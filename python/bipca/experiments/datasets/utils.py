from pathlib import Path
from functools import reduce
import requests
import io
import time
from typing import Dict, List, Union, Optional, Callable, Any
from functools import singledispatch

import tasklogger
import pandas as pd
from anndata import AnnData
import pyarrow.csv as csv
import biomart

from bipca.experiments.types import T_AnnDataOrDictAnnData


def resolve_nested_inheritance(
    cls: type,
    attr: str,
    and_func: Optional[Callable] = None,
    reversed: bool = True,
):
    """Traverse the inheritance tree of a class, extracting non-abstract methods or
    attributes w/ deduplication. Finally, return a nested function in the specified
    direction of the inheritance tree.

    Parameters
    ----------
    cls
        The class to traverse.
    attr
        The attribute to extract.
    and_func
        A function to apply to the extracted attribute. If the function returns True,
        the attribute is kept. If the function returns False, the attribute is
        discarded.
    reversed
        Traverse the MRO in reverse order (parent to child)

    """

    if reversed:
        sl = slice(None, None, -1)
    else:
        sl = slice(None, None, 1)
    if and_func is None:

        def and_func(*args, **kwargs):
            return True

    else:
        assert isinstance(and_func, Callable), "and_func must be callable."

    accumulated_attrs = []

    for base in cls.__mro__[sl]:
        if attr_val := getattr(base, attr, False):
            if and_func(attr_val):
                accumulated_attrs.append(attr_val)
    return accumulated_attrs


def accepts_partial_downloads(header: requests.structures.CaseInsensitiveDict) -> bool:
    if header.get("Accept-Ranges", "foo").lower() == "bytes":
        return True
    return False


def download_url(
    url: str,
    path: Union[str, Path],
    chunk_size: int = 1024 * 1024,
    logger: Optional[tasklogger.TaskLogger] = None,
) -> bool:
    logger = (
        tasklogger.TaskLogger(level=0, if_exists="ignore") if logger is None else logger
    )
    path = Path(path).resolve()
    user_agent = (
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) "
        "Gecko/2009021910 Firefox/3.0.7"
    )
    request_headers = {
        "user-agent": user_agent,
    }
    # get the headers of the url
    with requests.get(url, headers=request_headers, stream=True) as r:
        r.raise_for_status()
        r_headers = r.headers

    if path.exists():
        current_size = path.stat().st_size
    else:
        path.parents[0].mkdir(parents=True, exist_ok=True)
        current_size = 0

    if accepts_partial_downloads(r_headers):
        request_headers["Range"] = f"bytes={current_size}-"

    # check the transfer encoding for chunking.
    chunk_size = (
        None if r.headers.get("Transfer-Encoding", False) == "chunked" else chunk_size
    )

    download = True
    if content_length := r.headers.get("Content-Length", False):
        # validate that we need to download
        content_length = int(content_length)
        content_length = content_length - current_size
        download = content_length > 0

    if not download:
        if path.exists():
            logger.log_info(f"{str(path)} exists and is complete size.")
            return True
        else:
            return False
    if download:
        with requests.get(url, headers=request_headers, stream=True) as r:
            try:
                r.raise_for_status()
                with io.BytesIO() as f:
                    progress = last_progress = 0
                    start_time = last_time = time.time()
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

                            # progress += len(chunk)
                            # now_time = time.time()
                            # if now_time - last_time > 0.5:
                            #     avg_rate = (progress / 1024**2) / (
                            #         now_time - start_time
                            #     )
                            #     instant_rate = (
                            #         (progress - last_progress)
                            #         / 1024**2
                            #         / (now_time - last_time)
                            #     )
                            #     if content_length:
                            #         cont_str = f"{content_length / 1024**2 :.2f}"
                            #         cont_str = f"of {cont_str}MB"
                            #     else:
                            #         cont_str = ""
                            #     logger.log_info(
                            #         f"Downloaded {progress / 1024**2 :.2f}MB "
                            #         f"{cont_str} in "
                            #         f"{now_time-start_time:.2f}s at "
                            #         f"{avg_rate:.2f}MB/s. "
                            #         f"Instantaneous rate: "
                            #         f"{instant_rate:.2f}MB/s"
                            #     )
                            #     last_progress = progress
                            #     last_time = now_time
                    path.write_bytes(f.getbuffer())

            except requests.exceptions.HTTPError as exc:
                if "416 Client Error" in str(exc) and path.exists():
                    logger.log_warning(
                        f"HTTP 416 Client Error and {path} exists. "
                        f"If there are issues accessing {path}, remove the file "
                        "and try again."
                    )
                else:
                    raise exc

    return True


def download_urls(
    urls: Dict[Union[str, Path], str],
    logger: Optional[tasklogger.TaskLogger] = None,
    **kwargs,
):
    logger = (
        tasklogger.TaskLogger(level=0, if_exists="ignore") if logger is None else logger
    )

    # urls is a dict of form {filename:url}
    for filename, url in urls.items():
        download_url(url, filename, logger=logger, **kwargs)


## File I/O
@singledispatch
def write_adata(
    adata: T_AnnDataOrDictAnnData, path: Union[Path, str], name: Optional[str] = None
):
    raise NotImplementedError(f"Cannot write type {type(adata)}")


@write_adata.register(AnnData)
def _(adata: AnnData, path: Union[Path, str], name: Optional[str] = None):
    if name is None:
        path = Path(path)
    else:
        path = Path(path) / str(name)
    if ".h5ad" not in str(path):
        path = Path(str(path) + ".h5ad")
    path.parents[0].mkdir(parents=True, exist_ok=True)
    adata.write(str(path))


@write_adata.register(dict)
def _(adata: Dict[str, AnnData], path: Union[Path, str], name: Optional[str] = None):
    # ignores name
    for k, val in adata.items():
        write_adata(val, path, name=k)


def read_csv_pyarrow_bad_colnames(
    fname: str,
    delimiter: str = ",",
    index_col: Union[int, None] = 0,
    logger: Optional[tasklogger.TaskLogger] = None,
) -> pd.DataFrame:
    """csv_get_corrected_column_names: Read a csv file with fewer column names than
    columns. Return a DataFrame

    Used when the index column in a csv is unlabeled.

    Parameters
    ----------
    fname : str
        Path to file.
    delim : str, optional
        File delimiter, by default ","
    index_col : int or None, default 0
        Which column to index the output DataFrame by.
    Returns
    -------
    DataFrame
        DataFrame from csv
    """
    logger = (
        tasklogger.TaskLogger(level=0, if_exists="ignore") if logger is None else logger
    )
    fname = Path(fname).resolve()
    with logger.log_task(f"reading {fname.name} to DataFrame"):
        with open(fname, "r") as f:
            # extract the first two lines so we can compare and fix.
            l1 = f.readline()
            l2 = f.readline()
            # get the rest of the data
        # fix the missing columns
        l1 = l1.replace("\n", "")
        l1 = l1.replace('"', "")  # strip string characters
        l1 = l1.replace("\ufeff", "")  # strip the encoding character
        l2 = l2.replace("\n", "")

        l1 = l1.split(delimiter)
        l2 = l2.split(delimiter)

        if (ncols := len(l2) - len(l1)) > 0:
            l1 = [f"column{i}" for i in range(ncols)] + l1
        col_names = l1
        parse_options = csv.ParseOptions(delimiter=delimiter)
        base_block_size = 2**20 * 10  # 10mb at a time

        while True:
            try:
                read_options = csv.ReadOptions(
                    column_names=col_names, skip_rows=1, block_size=base_block_size - 1
                )
            except OverflowError as exc:
                raise exc

            try:
                tbl = csv.read_csv(
                    str(fname), read_options=read_options, parse_options=parse_options
                )
                break
            except Exception as exc:
                if (
                    "either file is too short or header is larger than block size"
                    in str(exc)
                ):
                    base_block_size *= 2
                else:
                    raise exc
        df = tbl.to_pandas()
        if index_col is not None:
            df = df.set_index(col_names[index_col])
    return df


## Biology tools
def get_ensembl_mappings(
    query_list: List[str],
    ensembl_server="http://jan2019.archive.ensembl.org/biomart",
    dataset="hsapiens_gene_ensembl",
    logger=None,
) -> Dict[str, Dict]:
    logger = (
        tasklogger.TaskLogger(level=0, if_exists="ignore") if logger is None else logger
    )
    # Set up connection to server
    with logger.log_task("ensembl genes annotations"):
        with logger.log_task(f"connecting to {ensembl_server}"):
            server = biomart.BiomartServer(ensembl_server)

        mart = server.datasets[dataset]

        # List the types of data we want
        attributes = [
            "ensembl_gene_id",
            "ensembl_transcript_id",
            "ensembl_peptide_id",
            "external_gene_name",
            "gene_biotype",
        ]

        # Get the mapping between the attributes
        with logger.log_task(f"biomart query"):
            try:
                response = mart.search(
                    {
                        "attributes": attributes,
                        "filters": {"ensembl_gene_id": query_list},
                    }
                )
            except requests.exceptions.HTTPError:
                response = mart.search({"attributes": attributes})
        ensembl_to_genesymbol = {}
        # Store the data in a dict
        for line in response.iter_lines():
            line = line.decode("utf-8")
            line = line.split("\t")
            # The entries are in the same order as in the `attributes` variable
            ensembl_gene = line[0]
            transcript_id = line[1]
            ensembl_peptide = line[2]
            gene_symbol = line[3]
            gene_biotype = line[4]

            # Some of these keys may be an empty string. If you want, you can
            # avoid having a '' key in your dict by ensuring the
            # transcript/gene/peptide ids have a nonzero length before
            # adding them to the dict
            ensembl_to_genesymbol[transcript_id] = {
                "gene_name": gene_symbol,
                "gene_biotype": gene_biotype,
            }
            ensembl_to_genesymbol[ensembl_gene] = {
                "gene_name": gene_symbol,
                "gene_biotype": gene_biotype,
            }
            ensembl_to_genesymbol[ensembl_peptide] = {
                "gene_name": gene_symbol,
                "gene_biotype": gene_biotype,
            }

    return {k: ensembl_to_genesymbol.get(k) for k in query_list}
