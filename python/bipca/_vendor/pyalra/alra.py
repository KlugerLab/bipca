import numpy as np
import numpy.typing as npt
import scipy as sp
from loguru import logger
from sklearn.utils.extmath import randomized_svd

from bipca._vendor.pyalra.choose_k import choose_k
from bipca._vendor.pyalra.logging import init_logger


def alra(
    a_norm: npt.ArrayLike | sp.sparse.spmatrix,
    k: int = 0,
    q: int = 10,
    quantile_prob: float = 0.001,
    seed: float | None = None,
    debug: bool = False,
    *,
    save_log: bool = False,
    log_level: int = 1,
) -> dict[str, npt.ArrayLike]:
    if debug:
        init_logger(log_level, save_log)

    logger.info(f"Read matrix with {a_norm.shape[0]} cells and {a_norm.shape[1]} genes")

    if k == 0:
        k, _, u, d, v = choose_k(a_norm)
        logger.info(f"Chose k = {k}")
    else:
        logger.debug("Running rsvd")
        u, d, v = randomized_svd(M=a_norm, n_components=k, n_iter=q, random_state=seed)

    logger.info("Getting nonzeros\n")
    originally_nonzero = a_norm > 0

    a_norm_rank_k = u @ np.diag(d) @ v

    logger.info(f"Find the {quantile_prob} quantile for each gene")
    a_norm_rank_k_mins = np.absolute(np.quantile(a_norm_rank_k, axis=0, q=0.001))

    logger.info("Sweep")
    a_norm_rank_k_cor = np.zeros(shape=a_norm_rank_k.shape)
    np.copyto(
        dst=a_norm_rank_k_cor,
        src=a_norm_rank_k,
        where=(a_norm_rank_k > np.tile(a_norm_rank_k_mins, (len(a_norm_rank_k), 1))),
    )

    sigma_1 = np.nanstd(a_norm_rank_k_cor, axis=0)
    sigma_2 = np.nanstd(a_norm, axis=0)
    mu_1 = np.divide(a_norm_rank_k_cor.sum(axis=0), a_norm_rank_k_cor.sum(axis=0))
    mu_2 = np.divide(a_norm.sum(axis=0), (a_norm > 0).sum(axis=0))

    toscale = np.logical_and(
        np.logical_and(~np.isnan(sigma_1), ~np.isnan(sigma_2)),
        np.logical_and(~np.logical_and((sigma_1 == 0), (sigma_2 == 0)), ~(sigma_1 == 0)),
    )

    logger.info(f"Scaling all except for {sum(~toscale)} columns")

    sigma_1_2 = np.divide(sigma_2, sigma_1)

    toadd = np.add(-np.divide(np.multiply(mu_1, sigma_2), sigma_1), mu_2)

    a_norm_rank_k_temp = a_norm_rank_k_cor[:, toscale].copy()
    a_norm_rank_k_temp = np.multiply(a_norm_rank_k_temp, sigma_1_2[toscale])
    a_norm_rank_k_temp = np.add(a_norm_rank_k_temp, toadd[toscale])

    a_norm_rank_k_cor_sc = a_norm_rank_k_cor.copy()
    np.copyto(dst=a_norm_rank_k_cor_sc[:, toscale], src=a_norm_rank_k_temp)
    a_norm_rank_k_cor_sc_zeros = np.zeros_like(a_norm_rank_k_cor_sc)
    np.copyto(dst=a_norm_rank_k_cor_sc, src=a_norm_rank_k_cor_sc_zeros, where=(a_norm_rank_k_cor == 0))

    lt0 = a_norm_rank_k_cor_sc < 0
    np.copyto(dst=a_norm_rank_k_cor_sc, src=a_norm_rank_k_cor_sc_zeros, where=lt0)

    a_norm_size = a_norm.shape[0] * a_norm.shape[1]
    logger.info(
        f"{100*np.sum(lt0)/a_norm_size:.2f}% of the values became negative in the scaling process and were set to zero"
    )

    nonzero_mask = np.logical_and(originally_nonzero, (a_norm_rank_k_cor_sc == 0))
    np.copyto(dst=a_norm_rank_k_cor_sc, src=a_norm, where=nonzero_mask)

    original_nz = np.sum(a_norm > 0) / a_norm_size
    completed_nz = np.sum(a_norm_rank_k_cor_sc > 0) / a_norm_size
    logger.info(f"The matrix went from {100*original_nz:.2f}% nonzero to {100*completed_nz:.2f}% nonzero")

    return {
        "A_norm_rank_k": a_norm_rank_k,
        "A_norm_rank_k_cor": a_norm_rank_k_cor,
        "A_norm_rank_k_cor_sc": a_norm_rank_k_cor_sc,
    }
