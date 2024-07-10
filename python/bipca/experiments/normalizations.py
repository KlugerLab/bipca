from numbers import Number
import os
import subprocess
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import tasklogger
from torch import log as tensor_log
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from sklearn.preprocessing import scale as zscore

from scanpy.experimental.pp import normalize_pearson_residuals
from anndata import AnnData, read_h5ad
import pyalra 
from pyalra.choose_k import choose_k
from pyalra.alra import alra

from bipca import BiPCA
from bipca.safe_basics import multiply,sum
from bipca.math import library_normalize
from bipca.base import log_func_with
from bipca.utils import is_tensor,issparse
from .utils import read_csv_pyarrow_bad_colnames





def log1p(A, scale="median", log_func=None):
    """log1p:
    Compute log1p transform commonly applied to single cell data. This procedure computes the sum along the rows of the data as the "library size".
    Then, the data is normalized so that the rows sum to 1. Next, a scale factor (by default the median of the library size) is multiplied by the normalized data. Finally,
    the natural log of 1 + the normalized and scaled data is computed.
    Parameters
    ----------
    A : array-like or AnnData
        The input data to process.
    scale : {numbers.Number, 'median'}, default 'median'
        The scale factor to apply to the data
    """
    def apply_log1p():
        if isinstance(A, AnnData):
            X = A.X
        else:
            X = A
        if scale != "median" and not isinstance(scale, Number):
            raise ValueError("`scale` must be 'median' or a Number")
        to_log = library_normalize(X, scale=scale)
        if is_tensor(X):
            return tensor_log(to_log + 1)
        else:
            if issparse(X):
                to_log.data += 1
                to_log.data = np.log(to_log.data)
                return to_log
            else:
                return np.log(to_log + 1)
    if log_func is None:
        return apply_log1p()
    else:
        return log_func_with(apply_log1p,log_func,'log1p normalization')()

_default_normalization_kwargs = {"log1p":{},
                            "log1p+z":{},
                            "Pearson":dict(clip=np.inf),
                            "ALRA":{"seed":42},
                            "Sanity":{}, 
                            "BiPCA":dict(n_components=-1, backend="torch", n_subsamples=0)}
def apply_normalizations(
    adata: Optional[AnnData] = None,
    read_path: Optional[Union[Path, str]] = None,
    write_path: Optional[Union[Path, str]] = None,
    n_threads=32,
    apply=["log1p", "log1p+z", "Pearson", "ALRA", "Sanity", "BiPCA"],
    normalization_kwargs = {},
    logger: Optional[tasklogger.TaskLogger] = None,
    sanity_installation_path: Union[Path, str] = "/docker/Sanity/bin/Sanity",
    sanity_tmp_path: Optional[Union[Path, str]] = None,
    recompute=False
):
    """
    Apply various normalization methods to the input data.

    Parameters
    ----------
    adata : AnnData, optional
        An AnnData object containing the data to normalize. If not provided, data will be read from `read_path`.
    read_path : Union[Path, str], optional
        Path to the input data file. Required if `adata` is not provided.
    write_path : Union[Path, str], optional
        Path to write the normalized data. If not provided, the normalized data will not be saved.
    n_threads : int, default 32
        Number of threads to use for computations.
    apply : list, default ["log1p", "log1p+z", "Pearson", "ALRA", "Sanity", "BiPCA"]
        List of normalization methods to apply. Can include custom callable functions.
    normalization_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the normalization methods.
    logger : tasklogger.TaskLogger, optional
        A TaskLogger object for logging progress. If not provided, a new TaskLogger will be created.
    sanity_installation_path : Union[Path, str], default "/docker/Sanity/bin/Sanity"
        Path to the Sanity binary.
    sanity_tmp_path : Union[Path, str], optional
        Path to a temporary directory for storing intermediate files from Sanity. If not provided, a 'tmp' directory in the current working directory will be used.
    recompute : bool, default False
        If True, recompute all normalizations even if they already exist in `adata`.

    Returns
    -------
    adata : AnnData
        The input AnnData object with additional layers for each applied normalization method.
    """
    logger = (
        tasklogger.TaskLogger(level=0, if_exists="ignore") if logger is None else logger
    )

   
    for key in apply:
        if key not in ['log1p','log1p+z', 'Pearson', 'ALRA', 'Sanity', 'BiPCA'] and \
            not callable(key):
            raise ValueError(f"Unknown normalization method {key}")

    
    #load the data
    if isinstance(adata, AnnData):
        pass
    else:
        if adata is not None:
            raise ValueError("adata must be an AnnData object")
        
    # Read data
    if adata is None:
        if read_path is None:
            if write_path is not None:
                read_path = write_path
            else:
                raise ValueError("read_path must be specified if adata is not specified.")
        if not read_path.exists():
            raise FileNotFoundError(f"{read_path} does not exist")
        with logger.log_task("loading count data for normalization"):
            adata = read_h5ad(read_path)
    xlog1p = None
    if any([ele in apply for ele in ['log1p', 'log1p+z','ALRA']]):
        #these are methods that require log1p, so we precompute this.
        if 'log1p' in adata.layers and not recompute:
            xlog1p = adata.layers["log1p"]
        else:
            xlog1p=log1p(adata.X,log_func=logger.log_task)
            if 'log1p' in apply:
                adata.layers["log1p"] = xlog1p
            else:
                pass
    

    # now apply remaining methods
    method_applied=False
    for method in filter(lambda x: x != 'log1p', apply): #skip log1p because it should already be handled
        if method == 'BiPCA':
            #check if Y_biwhite and Z_biwhite are in adata
            if 'Y_biwhite' in adata.layers and 'Z_biwhite' in adata.layers and not recompute:
                continue
        if method in adata.layers and not recompute:
            continue
        with logger.log_task(f"Applying {method} normalization"):
            method_applied=True
            current_kwargs = normalization_kwargs.get(method,
                                                      _default_normalization_kwargs.get(method,{}))
            match method:
                case "log1p+z":
                    if issparse(xlog1p):
                        adata.layers["log1p+z"] = zscore(xlog1p.toarray())
                    else:
                        adata.layers["log1p+z"] = zscore(xlog1p)
                case "Pearson":
                    result_dict = normalize_pearson_residuals(
                                    adata, inplace=False, **current_kwargs
                                    )
                    adata.layers["Pearson"] = result_dict["X"]
                case "ALRA":
                    # record the rank ALRA chooses
                    alra_k,*_ = choose_k(xlog1p,**current_kwargs)
                    adata.uns["ALRA"] = {"alra_k":alra_k}
                    adata.layers["ALRA"] = alra(xlog1p,k=adata.uns["ALRA"]["alra_k"], **current_kwargs)['A_norm_rank_k_cor_sc']
                case "BiPCA":
                    if "logger" not in current_kwargs:
                        current_kwargs["logger"] = logger
                    op = BiPCA(**current_kwargs).fit(adata)
                    op.get_plotting_spectrum()
                    op.write_to_adata(adata)
                case "Sanity":
                    if issparse(adata.X):
                        X = adata.X
                    else:
                        X = csr_matrix(adata.X)
                    # Specify the temporary folder that will store the output from intermediate outputs from Sanity
                    if sanity_tmp_path is None:
                        if write_path is not None:
                            sanity_tmp_path = Path(write_path).parent / "tmp"
                        elif read_path is not None:
                            sanity_tmp_path = Path(read_path).parent / "tmp"
                        else:
                            # get the current working directory
                            sanity_tmp_path = Path(os.getcwd()) / "tmp"
                    sanity_tmp_path.mkdir(parents=True, exist_ok=True)
                    # write intermediate files from sanity
                    sanity_counts_path = sanity_tmp_path / "count.mtx"
                    sanity_cells_path = sanity_tmp_path / "barcodes.tsv"
                    sanity_genes_path = sanity_tmp_path / "genes.tsv"
                    mmwrite(str(sanity_counts_path), X.T)
                    pd.Series(adata.obs_names).to_csv(
                        sanity_cells_path,
                        sep="\t",
                        index=False,
                        header=None,
                    )
                    pd.Series(adata.var_names).to_csv(
                        sanity_genes_path, sep="\t", index=False, header=None
                    )
                    sanity_command = (
                        sanity_installation_path
                        + " -f "
                        + str(sanity_counts_path)
                        + " -mtx_genes "
                        + str(sanity_genes_path)
                        + " -mtx_cells "
                        + str(sanity_cells_path)
                        + " -d "
                        + str(sanity_tmp_path)
                        + " -n "
                        + str(n_threads)
                    )
                    subprocess.run(sanity_command.split())
                    sanity_output = sanity_tmp_path / "log_transcription_quotients.txt"
                    adata.layers["Sanity"] = (
                        read_csv_pyarrow_bad_colnames(sanity_output,
                            delimiter="\t",index_col=0).reindex(adata.var_names)
                            .to_numpy().T
                        )
                case _:
                    adata.layers[getattr(method, '__name__', repr(method))] = \
                        method(adata.X,**current_kwargs)
                

    if write_path is not None and method_applied:
        with logger.log_task("writing normalized data"):
            adata.write(write_path)
    return adata