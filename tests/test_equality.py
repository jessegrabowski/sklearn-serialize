import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sklearn_serialize.equality import nested_equal


class TestNanEquality:
    """
    NaN == NaN is the primary reason nested_equal exists — Python's == and
    numpy's array_equal do not consider NaN equal to itself by default.
    """

    def test_float_nan(self):
        assert nested_equal(float("nan"), float("nan"))

    def test_np_float_nan(self):
        assert nested_equal(np.float64("nan"), np.float64("nan"))

    def test_nan_in_list_recurses(self):
        assert nested_equal([float("nan"), 1.0], [float("nan"), 1.0])

    def test_nan_in_dict_recurses(self):
        assert nested_equal({"x": float("nan")}, {"x": float("nan")})

    def test_nan_in_ndarray(self):
        a = np.array([1.0, float("nan"), 3.0])
        assert nested_equal(a, a.copy())

    def test_datetime64_nat_equals_nat(self):
        a = np.array(["2021-01-01", "NaT"], dtype="datetime64")
        assert nested_equal(a, a.copy())


class TestInfEquality:
    def test_float_inf(self):
        assert nested_equal(float("inf"), float("inf"))
        assert nested_equal(float("-inf"), float("-inf"))

    def test_np_float_inf(self):
        assert nested_equal(np.float64("inf"), np.float64("inf"))

    def test_float_inf_sign_mismatch(self):
        assert not nested_equal(float("inf"), float("-inf"))

    def test_np_float_inf_sign_mismatch(self):
        assert not nested_equal(np.float64("inf"), np.float64("-inf"))


class TestTypeMismatch:
    """
    Each dispatched type has an explicit isinstance guard. These verify those
    guards fire rather than silently comparing across incompatible types.
    """

    def test_float_vs_int(self):
        assert not nested_equal(1.0, 1)

    def test_np_float_vs_python_float(self):
        assert not nested_equal(np.float64(1.0), 1.0)

    def test_np_integer_vs_python_int(self):
        assert not nested_equal(np.int64(5), 5)

    def test_list_vs_tuple(self):
        assert not nested_equal([1, 2], (1, 2))

    def test_dict_vs_list_of_pairs(self):
        assert not nested_equal({"a": 1}, [("a", 1)])

    def test_set_vs_frozenset(self):
        assert not nested_equal({1, 2}, frozenset({1, 2}))

    def test_ndarray_vs_list(self):
        assert not nested_equal(np.array([1, 2]), [1, 2])

    def test_sparse_vs_ndarray(self):
        assert not nested_equal(sparse.eye(3, format="csr"), np.eye(3))


class TestNumpyArray:
    def test_dtype_mismatch_returns_false(self):
        assert not nested_equal(np.array([1], dtype=np.int32), np.array([1], dtype=np.int64))

    def test_shape_mismatch_returns_false(self):
        assert not nested_equal(np.array([1, 2]), np.array([[1, 2]]))

    def test_datetime64_mismatch_returns_false(self):
        a = np.array(["2021-01-01"], dtype="datetime64")
        b = np.array(["2021-01-02"], dtype="datetime64")
        assert not nested_equal(a, b)


class TestSparse:
    def test_csr_equal(self):
        m = sparse.eye(3, format="csr")
        assert nested_equal(m, m.copy())

    def test_coo_equal(self):
        m = sparse.eye(3, format="coo")
        assert nested_equal(m, m.copy())

    def test_lil_equal(self):
        # lil falls through to the toarray() comparison branch
        m = sparse.eye(3, format="lil")
        assert nested_equal(m, m.copy())

    def test_format_mismatch_returns_false(self):
        assert not nested_equal(sparse.eye(3, format="csr"), sparse.eye(3, format="csc"))

    def test_shape_mismatch_returns_false(self):
        assert not nested_equal(sparse.eye(3, format="csr"), sparse.eye(4, format="csr"))

    def test_values_mismatch_returns_false(self):
        a = sparse.eye(3, format="csr")
        assert not nested_equal(a, a * 2)


class TestPandas:
    """nested_equal delegates to .equals(), which is NaN-safe unlike ==."""

    def test_dataframe_nan_safe(self):
        df = pd.DataFrame({"a": [1.0, float("nan")]})
        assert nested_equal(df, df.copy())

    def test_dataframe_not_equal(self):
        assert not nested_equal(
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"a": [1, 3]}),
        )

    def test_series_nan_safe(self):
        s = pd.Series([1.0, float("nan")])
        assert nested_equal(s, s.copy())
