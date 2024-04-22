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

import biomart

from bipca.experiments.types import T_AnnDataOrDictAnnData
from bipca.experiments.utils import download_url, download_urls,get_files, read_csv_pyarrow_bad_colnames

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




## File I/O
@singledispatch
def write_adata(
    adata: T_AnnDataOrDictAnnData, path: Union[Path, str], name: Optional[str] = None, 
    overwrite:bool=True
):
    raise NotImplementedError(f"Cannot write type {type(adata)}")


@write_adata.register(AnnData)
def _(adata: AnnData, path: Union[Path, str], name: Optional[str] = None,
    overwrite:bool=True
):
    if name is None:
        path = Path(path)
    else:
        path = Path(path) / str(name)
    if ".h5ad" not in str(path):
        path = Path(str(path) + ".h5ad")
    path.parents[0].mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    if overwrite or not path.exists():
        adata.write(str(path))


@write_adata.register(dict)
def _(adata: Dict[str, AnnData], path: Union[Path, str], name: Optional[str] = None,
    overwrite:bool=True
    ):
    # ignores name
    for k, val in adata.items():
        write_adata(val, path, name=k)





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
