"""Summary
"""
from functools import singledispatch
import numpy as np
import inspect
import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from collections.abc import Iterable
from itertools import count
import tasklogger
from sklearn.base import BaseEstimator
from sklearn import set_config
import torch


### Functions for type checking etc
def is_valid(condition=lambda x: True, name=None, value=None):
    if not condition(value):
        raise ValueError(f"value for {name!r} is invalid.")


###Some functions the user may want


def write_to_adata(obj, adata):
    """Store the main outputs of a bipca object in place into an AnnData object.
        Note that if `obj.conserve_memory is True`, then adata.layers['Z'] cannot be written directly.

    Parameters
    ----------
    obj : bipca.BiPCA
        BiPCA object to extract results from
    adata : AnnData
        AnnData object to store results in.

    Returns
    -------
    adata : AnnData
        The modified adata object.

    Raises
    ------
    ValueError

    """
    target_shape = adata.shape
    if target_shape != (obj.M, obj.N) and target_shape != (obj.N, obj.M):
        raise ValueError(
            "Invalid shape passed. Adata must have shape "
            + str((obj.M, obj.N))
            + " or "
            + str((obj.N, obj.M))
        )
    with obj.logger.task("Writing bipca to anndata"):
        try:
            adata.uns["bipca"]
        except KeyError as e:
            adata.uns["bipca"] = {}

        try:
            if obj.conserve_memory:
                # the biwhitened data and the denoised matrix are NOT stored.
                pass
            else:
                adata.layers["Y_biwhite"] = make_scipy(obj.Y)
                adata.layers["Z_biwhite"] = make_scipy(
                    obj.transform(unscale=False, counts=True)
                )
        except Exception as err:
            raise Exception("Unable to write matrices to adata.") from err

        adata.varm["V_biwhite"] = make_scipy(obj.V_Y)
        adata.obsm["U_biwhite"] = make_scipy(obj.U_Y)

        adata.uns["bipca"]["left_biwhite"] = make_scipy(obj.left_biwhite)
        adata.uns["bipca"]["right_biwhite"] = make_scipy(obj.right_biwhite)

        adata.uns["bipca"]["S"] = make_scipy(obj.S_Y)
        adata.uns["bipca"]["rank"] = obj.mp_rank
        adata.uns["bipca"]["variance_estimator"] = obj.variance_estimator
        try:
            adata.uns["bipca"]["fits"]["kst"] = make_scipy(obj.kst)
            adata.uns["bipca"]["fits"]["kst_pvals"] = make_scipy(obj.kst_pvals)
        except:
            pass
        try:
            adata.uns["bipca"]["plotting_spectrum"] = obj.plotting_spectrum
        except:
            pass
    return adata


def fill_missing(X):
    if sparse.issparse(X):
        typ = type(X)
        X = sparse.coo_matrix(X)
        missing_entries = np.isnan(X.data)
        rows = X.row[missing_entries]
        cols = X.col[missing_entries]
        X.data[missing_entries] = 0
        observed_entries = np.ones_like(X)
        observed_entries[rows, cols] = 0
        X.eliminate_zeros()
        X = typ(X)
    else:
        missing_entries = np.isnan(X)
        observed_entries = np.ones_like(X)
        observed_entries[missing_entries] = 0
        X = np.where(missing_entries, 0, X)
    return X


def _is_vector(x):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return x.ndim == 1 or x.shape[0] == 1 or x.shape[1] == 1


def _xor(lst, obj):
    """Summary

    Parameters
    ----------
    lst : TYPE
        Description
    obj : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    condeval = [ele == obj for ele in lst]
    condeval = sum(condeval)
    return condeval == 1


def zero_pad_vec(nparray, final_length):
    """Summary

    Parameters
    ----------
    nparray : TYPE
        Description
    final_length : TYPE
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    ValueError
        Description
    """
    # pad a vector (nparray) to have length final_length by adding zeros
    # adds to the largest axis of the vector if it has 2 dimensions
    # requires the input to have 1 dimension or at least one dimension has length 1.
    if (nparray.shape[0]) == final_length:
        z = nparray
    else:
        axis = np.argmax(nparray.shape)
        pad = final_length - nparray.shape[axis]
        if nparray.ndim > 1:
            if not 1 in nparray.shape:
                raise ValueError("Input nparray is not a vector")
        padshape = list(nparray.shape)
        padshape[axis] = pad
        z = np.concatenate((nparray, np.zeros(padshape)), axis=axis)
    return z


#### PANDAS CONVENIENCE FUNCTIONS


def relabel_categories(column, map_dict={}):
    """Stably relabel the values in a pandas column

    Parameters
    ----------
    column : pd.Series
        A series to rename
    map_dict : dict
        A dictionary of {old_value:new_value} mappings

    Returns
    -------
    pd.Series
        A pandas series of the same dtype as `column`, with the mappings replaced using map_dict

    """
    if column.dtype == "category":
        old_vals = column.cat.categories
    else:
        old_vals = column.unique()
    for c in old_vals:
        if c not in map_dict.keys():
            map_dict[c] = c

    column = column.map(map_dict).astype(str(column.dtype))
    return column


def filter_dict(dict_to_filter, thing_with_kwargs, negate=False):
    """
    Modified from
    https://stackoverflow.com/a/44052550
    User "Adviendha"

    Parameters
    ----------
    dict_to_filter : TYPE
        Description
    thing_with_kwargs : TYPE
        Description
    negate : bool, optional
        Description

    Returns
    -------
    TYPE
        Description

    """
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
        and param.name in dict_to_filter.keys()
    ]
    if negate:
        filtered_dict = {
            key: dict_to_filter[key]
            for key in dict_to_filter.keys()
            if key not in filter_keys
        }
    else:
        filtered_dict = {
            filter_key: dict_to_filter[filter_key] for filter_key in filter_keys
        }
    return filtered_dict


def ischanged_dict(old_dict, new_dict, keys_ignore=[]):
    """Summary

    Parameters
    ----------
    old_dict : TYPE
        Description
    new_dict : TYPE
        Description
    keys_ignore : list, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    ischanged = False

    # check for adding or updating arguments
    for k in new_dict:
        ischanged = (
            k not in old_dict or old_dict[k] != new_dict[k]
        )  # catch values that are new
        if ischanged:
            break
    # now check for removing arguments
    if not ischanged:
        for k in old_dict:
            if k not in keys_ignore and k not in new_dict:
                ischanged = True
                break
    return ischanged


def issparse(X, check_torch=True, check_scipy=True):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description
    check_torch : bool, optional
        Description
    check_scipy : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    # Checks if X is a sparse tensor or matrix
    # returns False if not sparse
    # if sc
    if check_torch:
        if is_tensor(X):
            return "sparse" in str(X.layout)
    if check_scipy:
        return sparse.issparse(X)

    return False


def is_tensor(X):
    return isinstance(X, torch.Tensor)


def is_scipy(X):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    valid_X = [isinstance(X, np.ndarray), sparse.issparse(X)]
    if not any(valid_X):
        return False
    else:
        return True


def attr_exists_not_none(obj, attr):
    """Summary

    Parameters
    ----------
    obj : TYPE
        Description
    attr : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return hasattr(obj, attr) and not getattr(obj, attr) is None


def make_tensor(X, keep_sparse=True):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description
    keep_sparse : bool, optional
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    TypeError
        Description
    """
    if sparse.issparse(X):
        if keep_sparse:
            coo = X.tocoo()
            values = coo.data
            indices = np.vstack((coo.row, coo.col))
            i = torch.LongTensor(indices)
            v = torch.DoubleTensor(values)
            shape = coo.shape
            y = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
            y = y.coalesce()
            if sparse.isspmatrix_csr(X):
                y = y.to_sparse_csr()
        else:
            y = torch.from_numpy(X.toarray()).double()
    elif isinstance(X, np.ndarray):
        y = torch.from_numpy(X).double()
    elif is_tensor(X):
        y = X
    else:
        raise TypeError("Input matrix x is not sparse," + "np.array, or a torch tensor")
    return y


def make_scipy(X, keep_sparse=True):
    """Summary

    Parameters
    ----------
    X : TYPE
        Description
    keep_sparse : bool, optional
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    TypeError
        Description
    """
    if is_scipy(X):
        return X
    elif is_tensor(X):
        if issparse(X):
            shp = tuple(X.shape)
            if X.layout == torch.sparse_coo:
                if not X.is_coalesced():
                    X = X.coalesce()
                indices = X.indices().numpy()
                values = X.values().numpy()
                return sparse.coo_matrix((values, indices), shape=shp)
            elif X.layout == torch.sparse_csr:
                crow_indices = X.crow_indices().numpy()
                col_indices = X.col_indices().numpy()
                values = X.values().numpy()
                return sparse.csr_matrix((values, col_indices, crow_indices), shape=shp)
            else:
                raise ValueError("Unsupported sparse tensor input.")
        else:
            return X.numpy()


def stabilize_matrix(
    X,
    *,
    order=False,
    threshold=None,
    row_threshold=None,
    column_threshold=None,
    n_iters=0,
):
    """Filter the rows and/or columns of input matrix `mat` based on the number of
    nonzeros in each element

    Parameters
    ----------
    X : np.ndarray or scipy.spmatrix
        m x n input matrix
    order : bool or int, default False
        Perform filtering sequentially and specify order. Must be in [False, True, 0, 1].
        If False, filtering is performed using the original matrix.
        If True, perform filtering sequentially, starting with the rows.
        If integer, the integer specifies the first dimension to filter: 0 implies rows first,
            and 1 implies columns first.
    threshold : int, optional
        Global nonzero threshold for the rows and columns of the matrix.
        When `row_threshold` and `column_threshold` are not `None`,
        `threshold` is not used. Otherwise, sets the default condition for
        `row_threshold` and `column_threshold`
        If `threshold`, `row_threshold`, and `column_threshold` are None,
        defaults to 1.
    row_threshold, column_threshold  : int, optional
        Row (column) nonzero threshold of the matrix.
        Defaults to `threshold`. If `threshold` is `None`,
        defaults to 1.
    n_iters : int, default 0
        Repeat filtering for `n_iters` while the non-zero thresholds have not been met.
        If convergence is not reached, it is recommended to reconsider your stabilization,
        repeat with increased `n_iters`, or repeat starting from the preceding unconverged output.
    Returns
    -------
    Y : np.ndarray or scipy.spmatrix
        Output filtered matrix.
    indices : (np.ndarray(int), np.ndarray(int))
        Original indices in `X` used to produce `Y`,i.e.
        `X[indices[0],:][:,indices[1]] = Y`
    """
    if issparse(X, check_scipy=False):
        raise NotImplementedError("stabilize_matrix cannot be run on sparse tensors")
    if all([ele is None for ele in [threshold, row_threshold, column_threshold]]):
        threshold = 1
    if row_threshold is None:
        row_threshold = 1 if threshold is None else threshold
    if column_threshold is None:
        column_threshold = 1 if threshold is None else threshold
    assert (
        row_threshold is not None
    ), "`row_threshold` somehow is not set. Please file a bug report."
    assert (
        column_threshold is not None
    ), "`column_threshold` somehow is not set. Please file a bug report."
    assert order in [False, True, 0, 1], "`order` must be in [False, True, 0, 1]."

    if order is not False:
        first_dimension = 0 if order is True else order  # cast true to first dimension
        assert first_dimension in [0, 1]
        second_dimension = 1 - first_dimension
        indices = [np.ones([X.shape[0]], bool), np.ones([X.shape[1]], bool)]
        threshold = [row_threshold, column_threshold]
        indices[first_dimension] = (
            nz_along(X, axis=second_dimension) >= threshold[first_dimension]
        )
        Y = X[indices[0], :][:, indices[1]]
        indices[second_dimension] = (
            nz_along(Y, axis=first_dimension) >= threshold[second_dimension]
        )
        Y = X[indices[0], :][:, indices[1]]
    else:
        indices = [
            nz_along(X, axis=1) >= row_threshold,
            nz_along(X, axis=0) >= column_threshold,
        ]  # cols
        Y = X[indices[0], :]
        Y = Y[:, indices[1]]
    indices = [np.argwhere(ele).flatten() for ele in indices]

    if n_iters > 0:
        niters = n_iters
        assert isinstance(n_iters, int), "n_iters must be an integer"
        converged = lambda indices: all(
            [
                np.all(
                    nz_along(X[indices[0], :][:, indices[1]], axis=0)
                    >= column_threshold
                ),
                np.all(
                    nz_along(X[indices[0], :][:, indices[1]], axis=1) >= row_threshold
                ),
            ]
        )
        while not converged(indices) and n_iters > 0:
            n_iters -= 1
            Y, indices2 = stabilize_matrix(
                Y,
                order=order,
                threshold=threshold,
                row_threshold=row_threshold,
                column_threshold=column_threshold,
                n_iters=0,
            )
            indices = [
                inds0[indsnu] for inds0, indsnu in zip(indices, indices2)
            ]  # rebuild the indices
        if n_iters == 0 and not converged(indices):
            # didn't converge, recommend to user to increase n_iters.
            print(
                f"** Iterative filtering did not converge to target thresholds after {niters} iterations; "
                "inspect output Y and indices and consider repeating `stabilize_matrix`.\n"
                "\tTo start from the current filter, run \n"
                f"\t\t`Y2, indices2 = stabilize_matrix(X=Y, order={order}, threshold={threshold},"
                f"row_threshold={row_threshold}, column_threshold={column_threshold},"
                " n_iters=extra_iterations).\n\tRemap `indices2` to original indices by noting that"
                " Y2 = X[indices[0][indices2[0]],:][:,indices[1][indices2[1]]]"
            )
    return Y, indices


def spmatrix_count_nonzero(M, axis=0):
    if not issparse(M, check_torch=False):
        raise TypeError(
            "Input to `spmatrix_count_nonzero`" " must be scipy sparse matrix"
        )
    try:
        return M.getnnz(axis)
    except NotImplementedError as err:
        return M.tocsr().getnnz(axis)


def sptensor_count_nonzero(T, axis=0):
    # axis=0 moves down the rows, thus returning the number of nonzeros in each column,
    # while axis=1 moves over the columns, returning the number of nonzeros in each row.
    if not issparse(T, check_scipy=False):
        raise TypeError(
            "Input to `sptensor_count_nonzero`" " must be torch sparse tensor"
        )

    if T.layout == torch.sparse_csr:
        return sptensor_count_nonzero(T.to_sparse_coo(), axis=axis)
    else:
        # coo
        axis = abs(axis - 1)
        output = torch.zeros(T.shape[axis], dtype=int)
        if not T.is_coalesced():
            raise NotImplementedError(
                "sptensor_count_nonzero is not " "implemented for uncoalesced tensors."
            )
        else:
            # why tf does torch not have an eliminate_zeros method??
            valid_indices = T.indices()[:, T.values() != 0]
            inds, counts = valid_indices[axis, :].unique(return_counts=True)
            return output.scatter(0, inds, counts)


def nz_along(M, axis=0):
    """Count the nonzeros along an axis of a `scipy.sparse.spmatrix`, `torch.tensor`,
    or `numpy.ndarray`.


    Parameters
    ----------
    M : scipy.sparse.spmatrix or numpy.ndarray or torch.tensor
        M x N matrix to count the nonzeros in
    axis : int, default 0
        Axis to count nonzeros along. Follows numpy standard of directions:
        axis=0 moves down the rows, thus returning the number of nonzeros in each column,
        while axis=1 moves over the columns, returning the number of nonzeros in each row.

    Returns
    -------
    numpy.array
        Nonzeros along `axis` of `M`

    Raises
    ------
    TypeError
        Description
    ValueError
        Description
    TypeError
    ValueError
    """

    if not is_scipy(M) and not is_tensor(M):
        raise TypeError(
            "M must be an np.ndarray or scipy.spmatrix, not %s" % str(type(M))
        )
    iaxis = int(axis)
    if iaxis != axis:
        raise TypeError("axis must be an integer, not %s" % str(type(axis)))
    else:
        axis = iaxis

    ndim = M.ndim

    if axis < 0:
        axis = ndim + axis
    if axis > M.ndim - 1 or axis < 0:
        raise ValueError("axis out of range")

    if is_tensor(M):
        if issparse(M):
            countfun = lambda m: sptensor_count_nonzero(m, axis=axis).numpy()
        else:
            countfun = lambda m: torch.count_nonzero(m, dim=axis).numpy()
    else:
        if issparse(M):
            countfun = lambda m: spmatrix_count_nonzero(m, axis=axis)
        else:
            countfun = lambda m: np.count_nonzero(
                m, axis=axis
            )  # the number of nonzeros in each col
    return countfun(M)


def check_bound(other_dim, nzs, k=None):
    """Verify the upper bound in prop 6 from biwhitening paper.

    if nzs are an size-m vector containing the number of nonzeros on row 1,..,m, of an m x n matrix then other_dim is n.
    """
    if k == None:
        k = other_dim // 2
    zeros = other_dim - nzs
    n = len(nzs)
    assert k <= other_dim // 2
    p = other_dim - k
    bound = np.ceil((n * k) / other_dim)
    bad_dims = (zeros >= p).sum()
    truth = bad_dims < bound
    return truth, bad_dims


def auto_stabilize(X, step_size=None):
    # EXPERIMENTAL
    # stabilize the input matrix X according to the dimensional sparsity constraints in prop 6 of biwhitening paper.
    # BETA: should only work on ndarrays
    # also does not preserve original indices yet

    X, _ = stabilize_matrix(
        X, threshold=1
    )  # We need to make a matrix with no all-zero rows/columns. This step is what kills our indices
    mi, ni = X.shape
    # the vectors that keep track of the indices of the filtered submatrix.
    cols_kept = np.arange(ni)
    rows_kept = np.arange(mi)

    # build lists that keep track of the current nonzero locations along in each dimension
    # this is used instead of re-computing the nzs on every iteration
    nz_locs = np.argwhere(X)
    nz_locs_by_rows = np.split(
        nz_locs[:, 1], np.unique(nz_locs[:, 0], return_index=True)[1][1:]
    )
    nz_locs = nz_locs[nz_locs[:, 1].argsort()]
    nz_locs_by_cols = np.split(
        nz_locs[:, 0], np.unique(nz_locs[:, 1], return_index=True)[1][1:]
    )
    ell = k = 1

    # outer loop: iterate on ell and k (the bound for sparsity)
    while ell <= mi // 2 or k <= ni // 2:
        coli_passed = rowi_passed = False
        print(f"outer loop: ell={ell},k={k}")
        # inner loop: within a given ell and k, alternate between row and column removal until the bound is satisfied
        while not all([rowi_passed, coli_passed]):
            mi, ni = len(rows_kept), len(cols_kept)
            print(f"inner loop mi={mi},ni={ni}")
            # filter the columns
            if ell <= mi // 2:
                # get nnzs from the lists of nz locs
                col_nzi = np.asarray(
                    list(map(len, (nz_locs_by_cols[ix] for ix in cols_kept)))
                )
                mi, ni = len(rows_kept), len(cols_kept)
                coli_passed, ci = check_bound(
                    mi, col_nzi, k=ell
                )  # do the columns satisfy the bound? how many are in excess?
                if not coli_passed:
                    if step_size is None:
                        # use the tuned step-size according to the amount you'd need to remove to satisfy the bound on the next iteration
                        #  (i.e. updating the dimensions to a smaller submatrix)
                        # derived from letting m_(i+1) = mi-si and solving for si given that
                        # ri-si < ceil((mi-si) * k / ni)
                        si = np.maximum(
                            np.ceil((k * ni - mi * ci) / (k - mi)).astype(int), 1
                        )
                    else:
                        # use presupplied step size
                        si = step_size
                    cols_part = np.argpartition(
                        col_nzi, si
                    )  # sparsests columns from the submatrix..
                    cols_remove = cols_kept[cols_part[:si]]  # in real matrix indices
                    nz_locs_by_rows = [
                        ele[np.logical_not(np.in1d(ele, cols_remove))]
                        for ele in nz_locs_by_rows
                    ]
                    cols_kept = cols_kept[np.sort(cols_part[si:])]
                    rows_kept = rows_kept[
                        np.where(
                            np.fromiter(
                                map(len, (nz_locs_by_rows[ix] for ix in rows_kept)), int
                            )
                            > 0
                        )
                    ]  # remove empty rows from the row registry
            else:
                coli_passed = True

            # repeat the above, transposed for the rows!
            mi, ni = len(rows_kept), len(cols_kept)
            if k <= ni // 2:
                # get nnzs from the lists of nz locs

                row_nzi = np.asarray(
                    list(map(len, (nz_locs_by_rows[ix] for ix in rows_kept)))
                )

                rowi_passed, ri = check_bound(ni, row_nzi, k=k)
                if not rowi_passed:
                    if step_size is None:
                        si = np.maximum(
                            np.ceil((k * mi - ni * ri) / (k - ni)).astype(int), 1
                        )
                    else:
                        si = step_size
                    rows_part = np.argpartition(row_nzi, si)  # from the submatrix..
                    rows_remove = rows_kept[rows_part[:si]]  # in real matrix indices
                    nz_locs_by_cols = [
                        ele[np.logical_not(np.in1d(ele, rows_remove))]
                        for ele in nz_locs_by_cols
                    ]
                    rows_kept = rows_kept[np.sort(rows_part[si:])]
                    cols_kept = cols_kept[
                        np.where(
                            np.fromiter(
                                map(len, (nz_locs_by_cols[ix] for ix in cols_kept)), int
                            )
                            > 0
                        )
                    ]  # remove empty columns from the column registry
            else:
                rowi_passed = True
        # after filtering for the rows and columns, update all the bounds and parameters
        mi, ni = len(rows_kept), len(cols_kept)
        k += 1
        ell += 1

    # build the final output
    X = X[rows_kept, :][:, cols_kept]
    mi, ni = X.shape
    row_nz, col_nz = nz_along(X, axis=1), nz_along(X, axis=0)
    if all(map(lambda k: check_bound(ni, row_nz, k), range(ni // 2))) and all(
        map(lambda k: check_bound(mi, col_nz, k), range(mi // 2))
    ):
        return X  # we succeeded
    else:
        return auto_stabilize(X, step_size=step_size)  # bounds not satisfied, recur


def auto_stabilize2(X, step_size=None):
    # EXPERIMENTAL
    # stabilize the input matrix X according to the dimensional sparsity constraints in prop 6 of biwhitening paper.
    # this function differs from auto_stabilize as it only operates on the upper bound (m//2 and n//2)
    # BETA: should only work on ndarrays
    # also does not preserve original indices yet

    X, _ = stabilize_matrix(
        X, threshold=1
    )  # We need to make a matrix with no all-zero rows/columns. This step is what kills our indices
    mi, ni = X.shape
    # the vectors that keep track of the indices of the filtered submatrix.
    cols_kept = np.arange(ni)
    rows_kept = np.arange(mi)

    # build lists that keep track of the current nonzero locations along in each dimension
    # this is used instead of re-computing the nzs on every iteration
    nz_locs = np.argwhere(X)
    nz_locs_by_rows = np.split(
        nz_locs[:, 1], np.unique(nz_locs[:, 0], return_index=True)[1][1:]
    )
    nz_locs = nz_locs[nz_locs[:, 1].argsort()]
    nz_locs_by_cols = np.split(
        nz_locs[:, 0], np.unique(nz_locs[:, 1], return_index=True)[1][1:]
    )

    coli_passed = rowi_passed = False
    # inner loop: within a given ell and k, alternate between row and column removal until the bound is satisfied
    while not all([rowi_passed, coli_passed]):
        mi, ni = len(rows_kept), len(cols_kept)
        ell = mi // 2
        k = ni // 2
        print(f"inner loop mi={mi},ni={ni}")
        # filter the columns
        # get nnzs from the lists of nz locs
        col_nzi = np.asarray(list(map(len, (nz_locs_by_cols[ix] for ix in cols_kept))))
        print(col_nzi.min())

        mi, ni = len(rows_kept), len(cols_kept)
        coli_passed, ci = check_bound(
            mi, col_nzi, k=ell
        )  # do the columns satisfy the bound? how many are in excess?
        if not coli_passed:
            if step_size is None:
                # use the tuned step-size according to the amount you'd need to remove to satisfy the bound on the next iteration
                #  (i.e. updating the dimensions to a smaller submatrix)
                # derived from letting m_(i+1) = mi-si and solving for si given that
                # ri-si < ceil((mi-si) * k / ni)
                si = np.maximum(
                    np.ceil((k * ni - mi * ci) / (k - mi) * 0.01).astype(int), 1
                )
            else:
                # use presupplied step size
                si = step_size
            cols_part = np.argpartition(
                col_nzi, si
            )  # sparsests columns from the submatrix..
            cols_remove = cols_kept[cols_part[:si]]  # in real matrix indices
            nz_locs_by_rows = [
                ele[np.logical_not(np.in1d(ele, cols_remove))]
                for ele in nz_locs_by_rows
            ]
            cols_kept = cols_kept[np.sort(cols_part[si:])]
            rows_kept = rows_kept[
                np.where(
                    np.fromiter(
                        map(len, (nz_locs_by_rows[ix] for ix in rows_kept)), int
                    )
                    > 0
                )
            ]  # remove empty rows from the row registry

        # repeat the above, transposed for the rows!
        mi, ni = len(rows_kept), len(cols_kept)
        ell = mi // 2
        k = ni // 2
        # get nnzs from the lists of nz locs

        row_nzi = np.asarray(list(map(len, (nz_locs_by_rows[ix] for ix in rows_kept))))
        print(row_nzi.min())
        rowi_passed, ri = check_bound(ni, row_nzi, k=k)
        if not rowi_passed:
            if step_size is None:
                si = np.maximum(
                    np.ceil((k * mi - ni * ri) / (k - ni) * 0.01).astype(int), 1
                )
            else:
                si = step_size
            rows_part = np.argpartition(row_nzi, si)  # from the submatrix..
            rows_remove = rows_kept[rows_part[:si]]  # in real matrix indices
            nz_locs_by_cols = [
                ele[np.logical_not(np.in1d(ele, rows_remove))]
                for ele in nz_locs_by_cols
            ]
            rows_kept = rows_kept[np.sort(rows_part[si:])]
            cols_kept = cols_kept[
                np.where(
                    np.fromiter(
                        map(len, (nz_locs_by_cols[ix] for ix in cols_kept)), int
                    )
                    > 0
                )
            ]  # remove empty columns from the column registry

    # build the final output
    X = X[rows_kept, :][:, cols_kept]
    mi, ni = X.shape
    row_nz, col_nz = nz_along(X, axis=1), nz_along(X, axis=0)
    if check_bound(ni, row_nz, ni // 2) and check_bound(mi, col_nz, mi // 2):
        return X  # we succeeded
    else:
        return auto_stabilize2(X, step_size=step_size)  # bounds not satisfied, recur


def feature_scale(x, axis=-1):
    if axis == -1:
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        return (x - np.min(x, axis=axis)[:, None]) / (
            np.max(x, axis=axis)[:, None] - np.min(x, axis=axis)[:, None]
        )


class CachedFunction(object):
    def __init__(self, f, num_outs=1):
        self.f = f
        self.cache = {}
        self.num_outs = num_outs

    def compute_f(self, x):
        if isinstance(x, Iterable):
            x = tuple(x)
        if x in self.cache.keys():  # the value is already cached
            pass
        else:  # compute the value
            self.cache[x] = self.f(x)
        return self.cache[x]

    def keys(self):
        return self.cache.keys()

    def __call__(self, x):
        if isinstance(x, Iterable):
            if isinstance(x, np.ndarray):
                typef = lambda z: np.array(z)
            else:  # return eg a list
                typef = lambda z: type(x)(z)
            # evaluate the function
            # check that the number of outputs is stable
            outs = [[] for _ in range(self.num_outs)]
            for xx in x:
                y = self.compute_f(xx)
                if not (isinstance(y, Iterable)):
                    y = [y]
                if len(y) != self.num_outs:
                    raise ValueError(
                        "Number of outputs ({}) did not match ".format(len(y))
                        + "CachedFunction.num_outs ({})".format(self.num_outs)
                    )
                for yx in range(self.num_outs):
                    outs[yx] += [y[yx]]

            for lx in range(self.num_outs):
                outs[lx] = typef(outs[lx])
            outs = tuple(outs)
            if self.num_outs == 1:
                return outs[0]
            return outs

        return self.compute_f(x)


# Some "safe" math operations


def safe_all_non_negative(X):
    # check if a matrix is all non-negative in a type safe way
    # works for torch tensors, scipy sparse matrices, and numpy arrays
    if is_tensor(X):
        if issparse(X):  # sparse tensor
            return (X.values().min() >= 0).item()
        else:  # regular tensor
            return (X.min() >= 0).item()
    else:
        if issparse(X):  # sparse.spmatrix
            return X.data.min() >= 0
        else:  # np array
            return X.min() >= 0


def safe_elementwise_power(X, power=2):
    if is_tensor(X):
        if issparse(X):
            if X.layout == torch.sparse_csr:
                return torch.sparse_csr_tensor(
                    X.crow_indices(),
                    X.col_indices(),
                    torch.pow(X.values(), power),
                    X.size(),
                    dtype=X.dtype,
                )
            else:
                if X.is_coalesced():
                    return torch.sparse_coo_tensor(
                        X._indices(),
                        torch.pow(X.values(), power),
                        X.size(),
                        dtype=X.dtype,
                    ).coalesce()
                else:
                    return torch.sparse_coo_tensor(
                        X._indices(),
                        torch.pow(X.values(), power),
                        X.size(),
                        dtype=X.dtype,
                    )

        else:
            return torch.pow(X, power)
    else:
        if issparse(X):
            return X.power(power)
        else:
            return np.power(X, power)


def safe_elementwise_square(X):
    return safe_elementwise_power(X, 2)


def safe_hadamard(X, Y):
    # elementwise multiply the dimensionally compatible X * Y
    # where X or Y is a matrix
    # the output is coerced to be the same type of X.
    # designed for torch tensors, scipy sparse matrices, and numpy arrays

    if is_tensor(X):
        if issparse(X):
            raise NotImplementedError(
                "Safe hadamard not yet implemented for sparse tensors."
            )
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        return X * Y
    else:
        if issparse(X):
            return type(X)(X.multiply(Y))
        else:
            return np.multiply(X, Y)


def safe_dim_sum(X, dim=0, keep_type=False):
    # sum along a specified dimension
    # works for torch tensors, scipy sparse matrices, and numpy arrays
    # returns a numpy array unless keep_type=True
    if is_tensor(X):
        if issparse(X):  # sparse tensor
            s = torch_sparse_sum(X, dim=dim)
        else:  # regular tensor
            s = torch.sum(X, dim=dim)
        if not keep_type:
            s = s.numpy()

    else:
        s = np.asarray(X.sum(dim)).squeeze()

    return s


def safe_argsort(X, axis=-1, descending=False):
    if is_tensor(X):
        return torch.argsort(X, dim=axis, descending=descending)
    else:
        ix = np.argsort(X, axis=axis)
        if descending:
            return ix[::-1]
        else:
            return ix


def torch_sparse_sum(X, dim=0):
    assert dim == int(dim)  # accept only integer dimen
    dim = abs(dim - 1)
    if not is_tensor(X) or not issparse(X):
        raise ValueError("torch_sparse_sum only accepts sparse tensor inputs")
    out = torch.zeros(X.shape[dim], dtype=float)
    if X.layout == torch.sparse_coo:
        inds = X._indices()[dim, :]
        vals = X._values()
        return out.scatter(0, inds, vals, reduce="add").squeeze()
    else:
        # sparse csr
        # if dim==1:
        return torch_sparse_sum(X.to_sparse_coo(), dim=abs(dim - 1))


@singledispatch
def flatten(l):
    raise TypeError("Unsupported type {}".format(type(l)))


@flatten.register(list)
def flatten_list(l: list) -> list:
    """flatten_list flatten a list of lists of unknown depth into a single list"""

    l_out = []
    for item in l:
        if isinstance(item, list):
            l_out += flatten_list(item)
        else:
            l_out.append(item)
    return l_out
