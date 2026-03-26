import datetime
from collections.abc import Iterable, Sized
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def _nan_inf_equal(v1: Any, v2: Any) -> bool:
    # NaN != NaN by IEEE 754, but for round-trip equality we treat NaN == NaN.
    if np.isnan(v1) and np.isnan(v2):
        return True
    if np.isinf(v1) and np.isinf(v2):
        return (v1 > 0) == (v2 > 0)
    return v1 == v2


@singledispatch
def nested_equal(v1: Any, v2: Any) -> bool:
    """NaN-safe recursive equality for all types produced by json_to_data.

    Handles cases where == is insufficient: NaN != NaN in Python, numpy arrays
    require element-wise comparison, and pandas objects need .equals().

    Example
    -------
    >>> nested_equal(float("nan"), float("nan"))
    True
    >>> nested_equal(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
    True
    """
    return v1 == v2


@nested_equal.register(float)
def nested_equal_float(v1: float, v2: Any) -> bool:
    if not isinstance(v2, float):
        return False
    return _nan_inf_equal(v1, v2)


@nested_equal.register(np.floating)
def nested_equal_np_float(v1: np.floating, v2: Any) -> bool:
    if not isinstance(v2, np.floating):
        return False
    return _nan_inf_equal(v1, v2)


@nested_equal.register(np.integer)
def nested_equal_np_integer(v1: np.integer, v2: Any) -> bool:
    if not isinstance(v2, np.integer):
        return False
    return v1 == v2


@nested_equal.register(type)
def nested_equal_type(v1: type, v2: Any) -> bool:
    return v1 == v2


@nested_equal.register(str)
def nested_equal_str(v1: str, v2: Any) -> bool:
    if not isinstance(v2, str):
        return False
    return v1 == v2


@nested_equal.register(list)
def nested_equal_list(v1: list, v2: Any) -> bool:
    if not isinstance(v2, list):
        return False
    if len(v1) != len(v2):
        return False
    return all(nested_equal(item1, item2) for item1, item2 in zip(v1, v2, strict=False))


@nested_equal.register(tuple)
def nested_equal_tuple(v1: tuple, v2: Any) -> bool:
    if not isinstance(v2, tuple):
        return False
    if len(v1) != len(v2):
        return False
    return all(nested_equal(item1, item2) for item1, item2 in zip(v1, v2, strict=False))


@nested_equal.register(dict)
def nested_equal_dict(v1: dict, v2: Any) -> bool:
    if not isinstance(v2, dict):
        return False
    if v1.keys() != v2.keys():
        return False
    return all(nested_equal(v1[key], v2[key]) for key in v1)


@nested_equal.register(set)
def nested_equal_set(v1: set, v2: Any) -> bool:
    if not isinstance(v2, set):
        return False
    return v1 == v2


@nested_equal.register(frozenset)
def nested_equal_frozenset(v1: frozenset, v2: Any) -> bool:
    if not isinstance(v2, frozenset):
        return False
    return v1 == v2


@nested_equal.register(slice)
def nested_equal_slice(v1: slice, v2: Any) -> bool:
    if not isinstance(v2, slice):
        return False
    return (v1.start == v2.start) and (v1.stop == v2.stop) and (v1.step == v2.step)


@nested_equal.register(np.ndarray)
def nested_equal_ndarray(v1: np.ndarray, v2: Any) -> bool:
    if not isinstance(v2, np.ndarray):
        return False
    if v1.dtype != v2.dtype:
        return False
    if v1.shape != v2.shape:
        return False
    if np.issubdtype(v1.dtype, np.number):
        return np.array_equal(v1, v2, equal_nan=True)
    elif np.issubdtype(v1.dtype, np.datetime64):
        v1_isnat = np.isnat(v1)
        v2_isnat = np.isnat(v2)
        return bool(np.all((v1 == v2) | (v1_isnat & v2_isnat)))
    else:
        return np.array_equal(v1, v2)


@nested_equal.register(datetime.date)
def nested_equal_date(v1: datetime.date, v2: Any) -> bool:
    if not isinstance(v2, datetime.date):
        return False
    return v1 == v2


@nested_equal.register(Iterable)
def nested_equal_iterable(v1: Iterable, v2: Any) -> bool:
    if not isinstance(v2, Iterable):
        return False
    if isinstance(v1, str | bytes) or isinstance(v2, str | bytes):
        return v1 == v2
    if isinstance(v1, dict) or isinstance(v2, dict):
        return False
    if isinstance(v1, Sized) and isinstance(v2, Sized) and len(v1) != len(v2):
        return False
    return all(nested_equal(item1, item2) for item1, item2 in zip(v1, v2, strict=False))


@nested_equal.register(sparse.spmatrix)
def nested_equal_sparse(v1: sparse.spmatrix, v2: Any) -> bool:
    if (
        not isinstance(v2, sparse.spmatrix)
        or v1.getformat() != v2.getformat()
        or v1.shape != v2.shape
        or v1.nnz != v2.nnz
    ):
        return False
    if v1.getformat() in ["csr", "csc"]:
        return (
            np.array_equal(v1.data, v2.data)
            and np.array_equal(v1.indices, v2.indices)
            and np.array_equal(v1.indptr, v2.indptr)
        )
    elif v1.getformat() == "coo":
        return np.array_equal(v1.data, v2.data) and np.array_equal(v1.row, v2.row) and np.array_equal(v1.col, v2.col)
    else:
        return np.array_equal(v1.toarray(), v2.toarray())


@nested_equal.register(pd.DataFrame)
def nested_equal_pandas_dataframe(v1: pd.DataFrame, v2: Any) -> bool:
    return v1.equals(v2)


@nested_equal.register(pd.Series)
def nested_equal_pandas_series(v1: pd.Series, v2: Any) -> bool:
    return v1.equals(v2)
