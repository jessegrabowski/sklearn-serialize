"""
Round-trip tests: serialize → JSON string → restore must reproduce the original value.

Tests marked xfail document known bugs from the code audit. Remove the xfail
mark when the corresponding bug is fixed.
"""

import datetime
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sklearn_serialize.serialize._core as _serialize_core
from sklearn_serialize.serialize import data_to_json, json_to_data, trust_module


def roundtrip(data):
    return json_to_data(data_to_json(data))


class TestFloatSpecialValues:
    """float passthrough is fine; nan/inf require custom tagged encoding."""

    def test_nan(self):
        assert np.isnan(roundtrip(float("nan")))

    def test_pos_inf(self):
        assert roundtrip(float("inf")) == float("inf")

    def test_neg_inf(self):
        assert roundtrip(float("-inf")) == float("-inf")


class TestBytes:
    def test_utf8_roundtrip(self):
        assert roundtrip(b"hello world") == b"hello world"

    @pytest.mark.xfail(
        strict=True, reason="known bug: non-UTF8 bytes decoded with errors='replace', corrupting arbitrary binary data"
    )
    def test_binary_roundtrip(self):
        assert roundtrip(bytes([0xFF, 0xFE, 0x00])) == bytes([0xFF, 0xFE, 0x00])


class TestTuple:
    def test_roundtrip_as_tuple_not_list(self):
        # tuples get a custom tagged encoding — they must not come back as lists
        result = roundtrip((1, "a", None))
        assert result == (1, "a", None)
        assert isinstance(result, tuple)

    def test_nested_tuple(self):
        assert roundtrip(((1, 2), (3, 4))) == ((1, 2), (3, 4))


class TestNamedtuple:
    def test_type_and_values_preserved(self):
        Point = namedtuple("Point", ["x", "y"])
        result = roundtrip(Point(x=3, y=4))
        assert result == Point(3, 4)
        assert result._fields == ("x", "y")


class TestOrderedDict:
    def test_insertion_order_preserved(self):
        od = OrderedDict([("z", 1), ("a", 2), ("m", 3)])
        result = roundtrip(od)
        assert list(result.items()) == [("z", 1), ("a", 2), ("m", 3)]


class TestSet:
    def test_roundtrip_as_set_not_list(self):
        result = roundtrip({1, 2, 3})
        assert result == {1, 2, 3}
        assert isinstance(result, set)


class TestFrozenset:
    def test_roundtrip_as_frozenset(self):
        result = roundtrip(frozenset({1, 2, 3}))
        assert result == frozenset({1, 2, 3})
        assert isinstance(result, frozenset)


class TestSlice:
    def test_start_stop_step_preserved(self):
        assert roundtrip(slice(1, 10, 2)) == slice(1, 10, 2)

    def test_none_bounds_preserved(self):
        assert roundtrip(slice(None, 5)) == slice(None, 5)


class TestNumpyInt:
    def test_roundtrip(self):
        result = roundtrip(np.int64(42))
        assert result == 42
        assert isinstance(result, np.integer)

    @pytest.mark.xfail(strict=True, reason="known bug: all np.integer dtypes restored as np.int64, original dtype lost")
    def test_dtype_preserved(self):
        assert type(roundtrip(np.int32(42))) is np.int32


class TestNumpyFloat:
    def test_hex_encoding_preserves_precision(self):
        val = np.float64(1.0 / 3.0)
        assert roundtrip(val) == val

    def test_nan(self):
        assert np.isnan(roundtrip(np.float64("nan")))

    def test_inf(self):
        assert roundtrip(np.float64("inf")) == np.float64("inf")
        assert roundtrip(np.float64("-inf")) == np.float64("-inf")

    @pytest.mark.xfail(
        strict=True, reason="known bug: serializer never writes 'dtype' key, all np.floating restored as float64"
    )
    def test_float32_dtype_preserved(self):
        assert type(roundtrip(np.float32(1.5))) is np.float32


class TestNumpyType:
    def test_numpy_dtype_class_roundtrip(self):
        # numpy dtype classes appear as estimator params (e.g. dtype=np.float32)
        assert roundtrip(np.float32) is np.float32
        assert roundtrip(np.int64) is np.int64


class TestNumpyArray:
    def test_dtype_preserved(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        result = roundtrip(a)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, a)

    def test_float_nan_preserved(self):
        a = np.array([1.0, float("nan"), 3.0])
        result = roundtrip(a)
        assert np.array_equal(result, a, equal_nan=True)

    def test_shape_preserved(self):
        a = np.arange(6).reshape(2, 3)
        result = roundtrip(a)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, a)

    def test_datetime64_array(self):
        a = np.array(["2021-01-01", "2021-06-15"], dtype="datetime64[D]")
        result = roundtrip(a)
        np.testing.assert_array_equal(result, a)
        assert result.dtype == a.dtype

    def test_object_array_with_recursive_serialize(self):
        # object dtype triggers per-element serialize(); verifies recursive dispatch
        a = np.array([1, "text", None], dtype=object)
        result = roundtrip(a)
        np.testing.assert_array_equal(result, a)


class TestDatetime:
    def test_date(self):
        assert roundtrip(datetime.date(2021, 6, 15)) == datetime.date(2021, 6, 15)

    def test_datetime_with_microseconds(self):
        dt = datetime.datetime(2021, 6, 15, 12, 30, 45, 123456)
        assert roundtrip(dt) == dt

    def test_np_datetime64(self):
        d = np.datetime64("2021-01-01")
        assert roundtrip(d) == d


class TestSparse:
    def test_csr_roundtrip(self):
        m = sparse.eye(4, format="csr")
        result = roundtrip(m)
        assert result.format == "csr"
        assert (result - m).nnz == 0

    def test_csc_roundtrip(self):
        m = sparse.eye(4, format="csc")
        result = roundtrip(m)
        assert result.format == "csc"
        assert (result - m).nnz == 0

    def test_coo_roundtrip(self):
        m = sparse.eye(4, format="coo")
        result = roundtrip(m)
        assert result.format == "coo"
        np.testing.assert_array_equal(result.toarray(), m.toarray())

    def test_lil_roundtrip(self):
        m = sparse.eye(4, format="lil")
        result = roundtrip(m)
        assert result.format == "lil"
        np.testing.assert_array_equal(result.toarray(), m.toarray())

    def test_dok_roundtrip(self):
        m = sparse.eye(4, format="dok")
        result = roundtrip(m)
        assert result.format == "dok"
        np.testing.assert_array_equal(result.toarray(), m.toarray())


class TestPandasSeries:
    def test_float_series_with_nan(self):
        s = pd.Series([1.0, float("nan"), 3.0], name="value")
        pd.testing.assert_series_equal(roundtrip(s), s)

    def test_none_name_preserved(self):
        assert roundtrip(pd.Series([1, 2, 3])).name is None

    def test_integer_name(self):
        # np.int64 names are not JSON-serializable without going through serialize()
        s = pd.Series([1, 2, 3], name=np.int64(5))
        assert roundtrip(s).name == np.int64(5)

    def test_tuple_name(self):
        # tuple names appear in MultiIndex-derived Series; must not become a list
        s = pd.Series([1, 2, 3], name=(0, "level1"))
        result = roundtrip(s)
        assert result.name == (0, "level1")
        assert isinstance(result.name, tuple)


class TestPandasDataFrame:
    def test_mixed_dtypes_with_nan(self):
        df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.0, float("nan"), 3.0]})
        # check_dtype=False: pandas 3 infers all-integer float columns as int64
        pd.testing.assert_frame_equal(roundtrip(df), df, check_dtype=False)

    def test_datetime_columns_preserved(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-06-15", "2022-03-10"]),
                "val": [1.0, 2.0, 3.0],
            }
        )
        result = roundtrip(df)
        # check_dtype=False: pandas 3 infers integer-valued float columns as int64
        pd.testing.assert_frame_equal(result, df, check_dtype=False)
        assert result["date"].dtype == df["date"].dtype

    def test_categorical_columns_preserved(self):
        df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])})
        result = roundtrip(df)
        pd.testing.assert_frame_equal(result, df)
        assert result["cat"].dtype == "category"

    def test_ordered_categorical_ordering_preserved(self):
        cat = pd.Categorical(["low", "high", "low"], categories=["low", "mid", "high"], ordered=True)
        df = pd.DataFrame({"level": cat})
        result = roundtrip(df)
        pd.testing.assert_frame_equal(result, df)
        assert result["level"].cat.ordered


class TestSklearnEstimator:
    def test_constructor_params_preserved(self):
        scaler = StandardScaler(with_mean=False, with_std=True)
        result = roundtrip(scaler)
        assert result.with_mean is False
        assert result.with_std is True

    def test_fitted_attributes_preserved(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler().fit(X)
        result = roundtrip(scaler)
        np.testing.assert_array_equal(result.mean_, scaler.mean_)
        np.testing.assert_array_equal(result.scale_, scaler.scale_)

    def test_fitted_estimator_produces_identical_predictions(self):
        X = np.array([[1.0], [2.0], [3.0]])
        model = LinearRegression().fit(X, np.array([2.0, 4.0, 6.0]))
        np.testing.assert_array_almost_equal(roundtrip(model).predict(X), model.predict(X))

    def test_fitted_encoder_categories_preserved(self):
        enc = OneHotEncoder(sparse_output=False).fit(np.array([["a"], ["b"], ["c"]]))
        np.testing.assert_array_equal(roundtrip(enc).categories_[0], enc.categories_[0])


class TestPipeline:
    def test_step_names_and_types_preserved(self):
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        result = roundtrip(pipe)
        assert [name for name, _ in result.steps] == ["scaler", "lr"]
        assert isinstance(result.named_steps["scaler"], StandardScaler)

    def test_fitted_pipeline_produces_identical_predictions(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]).fit(X, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(roundtrip(pipe).predict(X), pipe.predict(X))


class TestFeatureUnion:
    def test_transformer_list_preserved(self):
        fu = FeatureUnion([("scaler", StandardScaler()), ("passthrough", "passthrough")])
        result = roundtrip(fu)
        assert [name for name, _ in result.transformer_list] == ["scaler", "passthrough"]


class TestTrustedModules:
    @pytest.fixture(autouse=True)
    def restore_trusted_modules(self, monkeypatch):
        monkeypatch.setattr(_serialize_core, "_TRUSTED_MODULES", _serialize_core._TRUSTED_MODULES.copy())

    def test_untrusted_module_raises(self):
        payload = data_to_json(StandardScaler().fit([[1], [2], [3]]))
        payload = payload.replace("sklearn.preprocessing", "malicious.package")
        with pytest.raises(ValueError, match="malicious.package"):
            json_to_data(payload)

    def test_trust_module_allows_deserialization(self):
        payload = data_to_json(StandardScaler().fit([[1], [2], [3]]))
        payload = payload.replace("sklearn.preprocessing", "mycompany.transformers")
        with pytest.raises(ValueError):
            json_to_data(payload)
        trust_module("mycompany")
        # now fails with ImportError (module doesn't exist), not ValueError
        with pytest.raises(ImportError):
            json_to_data(payload)

    def test_submodule_covered_by_prefix(self):
        # sklearn.preprocessing.* should be covered by the "sklearn" prefix
        scaler = StandardScaler().fit([[1], [2], [3]])
        assert roundtrip(scaler).mean_ is not None
