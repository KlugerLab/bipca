from numbers import Number
from typing import Any, Tuple, Optional, Union,Dict
from pathlib import Path
from urllib.parse import urlparse
from shutil import copyfile
import requests
import io
import time
import tasklogger
import pyarrow.csv as csv
import numpy as np
import pandas as pd
from .exceptions import handle_multiple_exceptions
from bipca.utils import flatten

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

def get_files(
        paths: Dict[str, str], download: bool = True, overwrite: bool = False,
        logger: Optional[tasklogger.TaskLogger] = None
    ):
    """ get_files: Download or copy files from a remote location to a local location.

    Parameters
    ----------
    paths : Dict[str, str]
        Dictionary of local paths to remote paths.
    download : bool, optional
        Whether to download files, by default True
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    logger : Optional[tasklogger.TaskLogger], optional
        Logger to use, by default None
    """
    logger = (
    tasklogger.TaskLogger(level=0, if_exists="ignore") if logger is None else logger)
    exceptions = {}
    to_download = {}
    to_copy = {}
    for local, remote in paths.items():
        local = str(local)
        remote = str(remote)
        if local == "." or remote == "." or local == "" or remote == "":
            exceptions[f"{local}: {remote}"] = IOError("Local path was '.'")
            continue
        else:
            local_path = Path(local).resolve()
            if overwrite or not local_path.exists():
                if (remote_path := Path(remote).resolve()).exists():
                    to_copy[local_path] = remote_path
                else:
                    if bool(urlparse(remote).scheme):
                        to_download[local_path] = remote
                if local_path not in to_copy and local_path not in to_download:
                    exceptions[f"{local_path.name}: {remote}"] = IOError(
                        "Invalid path or URL."
                    )
    for local_path, remote_path in to_copy.items():
        if not local_path == remote_path:
            with logger.log_task(f"copying {remote_path} to {local_path}"):
                try:
                    if not local_path.exists():
                        local_path.parents[0].mkdir(parents=True, exist_ok=True)
                    copyfile(remote_path, local_path)
                except Exception as e:
                    exceptions[f"{str(local_path.name)}: {str(remote)}"] = e
        else:
            logger.log_info(
                f"Skipping local path {local_path.name} because it"
                " matches the remote."
            )

    if to_download and download:
        for path, url in to_download.items():
            with logger.log_task(f"downloading {url}"):
                try:
                    download_url(url, path, logger=logger)
                except Exception as e:
                    exceptions[f"{str(local_path.name)}: {str(remote)}"] = e
    if exceptions:
        handle_multiple_exceptions(IOError, "Unable to acquire files: ", exceptions)

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

def parse_number_greater_than(
    number: Any,
    than: Number,
    name: str = None,
    equal_to: bool = False,
    typ: type = Number,
):
    if name is None:
        name = "input"
    if not isinstance(number, Number):
        raise TypeError(f"{name} must be a {typ}")
    if equal_to:
        if number < than:
            raise ValueError(f"{name} must be greater than or equal to {than}")
    else:
        if number <= than:
            raise ValueError(f"{name} must be greater than {than}")
    return number


def parse_number_less_than(
    number: Any,
    than: Number,
    name: str = None,
    equal_to: bool = False,
    typ: Union[type, Tuple[type]] = Number,
):
    if name is None:
        name = "input"
    if not isinstance(number, typ):
        raise TypeError(f"{name} must be a {typ}")
    if equal_to:
        if number > than:
            raise ValueError(f"{name} must be less than or equal to {than}")
    else:
        if number >= than:
            raise ValueError(f"{name} must be less than {than}")
    return number


def parse_mrows_ncols_rank(
    mrows: int, ncols: int, rank: Optional[int] = None
) -> Tuple[int, int, int]:
    """Parse the mrows, ncols, and rank arguments for random matrix functions"""
    mrows = parse_number_greater_than(
        mrows, 1, "mrows", equal_to=True, typ=(int, np.integer)
    )
    ncols = parse_number_greater_than(
        ncols, 1, "ncols", equal_to=True, typ=(int, np.integer)
    )
    if rank is not None:
        rank = parse_number_greater_than(
            rank, 1, "rank", equal_to=True, typ=(int, np.integer)
        )
        rank = parse_number_less_than(
            rank, min(mrows, ncols), "rank", equal_to=True, typ=(int, np.integer)
        )
    else:
        rank = None
    return mrows, ncols, rank


def get_rng(
    rng: Union[np.random._generator.Generator, Number]
) -> np.random._generator.Generator:
    """Parse the rng argument for random matrix functions"""
    if isinstance(rng, Number):
        rng = np.random.default_rng(rng)
    if not isinstance(rng, np.random._generator.Generator):
        raise TypeError("seed must be a Number or a numpy random generator")
    return rng


## General purpose tools
def uniques(it, key=None):
    seen = set()
    for x in it:
        if key is not None:
            xx = x[key]
        else:
            xx = x
        if xx not in seen:
            seen.add(xx)
            yield x
