import datetime
import math
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.ensemble import GradientBoostingRegressor
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

    def test_binary_roundtrip(self):
        assert roundtrip(bytes([0xFF, 0xFE, 0x00])) == bytes([0xFF, 0xFE, 0x00])

    def test_bytearray_roundtrip(self):
        assert roundtrip(bytearray(b"hello \x00\xff")) == bytearray(b"hello \x00\xff")


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


class TestNonStringKeyedDict:
    def test_tuple_key_restored(self):
        # tuple keys go through serialize(); without restore() on the key they come back as lists
        d = {(1, 2): "value"}
        result = roundtrip(d)
        assert result == {(1, 2): "value"}
        assert isinstance(next(iter(result)), tuple)

    def test_complex_value_restored(self):
        # values also need restore(); use a non-string key to force the py/dict path
        result = roundtrip({1: np.int64(42)})
        assert isinstance(result[1], np.integer)
        assert result[1] == 42


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

    def test_float32_dtype_preserved(self):
        assert type(roundtrip(np.float32(1.5))) is np.float32


class TestNumpyType:
    def test_numpy_dtype_class_roundtrip(self):
        # numpy dtype classes appear as estimator params (e.g. dtype=np.float32)
        assert roundtrip(np.float32) is np.float32
        assert roundtrip(np.int64) is np.int64

    def test_abstract_numpy_type_roundtrip(self):
        # Abstract types like np.integer are not dtype-constructable (np.dtype("integer") raises);
        # restore falls back to getattr(np, name) which handles all np.generic subclasses.
        assert roundtrip(np.integer) is np.integer
        assert roundtrip(np.floating) is np.floating


class TestComplex:
    def test_roundtrip(self):
        assert roundtrip(1 + 2j) == 1 + 2j

    def test_nan_inf(self):
        val = complex(float("nan"), float("inf"))
        result = roundtrip(val)
        assert isinstance(result, complex)
        assert math.isnan(result.real) and math.isinf(result.imag)


class TestNumpyComplex:
    def test_scalar_roundtrip(self):
        val = np.complex128(3.14 + 2.72j)
        result = roundtrip(val)
        assert result == val
        assert type(result) is np.complex128

    def test_complex64_dtype_preserved(self):
        val = np.complex64(1.0 + 2.0j)
        assert type(roundtrip(val)) is np.complex64

    def test_scalar_nan_inf(self):
        val = np.complex128(complex(float("nan"), float("inf")))
        result = roundtrip(val)
        assert np.isnan(result.real)
        assert np.isinf(result.imag)

    def test_array_roundtrip(self):
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        np.testing.assert_array_equal(roundtrip(a), a)

    def test_array_dtype_preserved(self):
        a = np.array([1 + 2j], dtype=np.complex64)
        assert roundtrip(a).dtype == np.complex64

    def test_2d_array_shape_preserved(self):
        a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        result = roundtrip(a)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, a)


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


class TestNumpyDtype:
    def test_numeric_dtype_roundtrip(self):
        for s in ["float64", "int32", "complex128", "bool"]:
            assert np.dtype(s) == roundtrip(np.dtype(s))


class TestGenerator:
    def test_state_preserved_after_roundtrip(self):
        rng = np.random.default_rng(42)
        rng.random(10)
        restored = roundtrip(rng)
        assert restored.random() == rng.random()

    def test_bit_generator_type_preserved(self):
        rng = np.random.default_rng(42)
        restored = roundtrip(rng)
        assert type(restored.bit_generator) is type(rng.bit_generator)


class TestRandomState:
    def test_state_preserved_after_roundtrip(self):
        rng = np.random.RandomState(42)
        rng.random(10)  # advance past the seed so we test serialised mid-stream state
        restored = roundtrip(rng)
        assert restored.random() == rng.random()


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


class TestPolarsDataFrame:
    def test_numeric_dtypes_with_nan(self):
        import polars as pl

        df = pl.DataFrame({"ints": [1, 2, 3], "floats": [1.0, float("nan"), 3.0]})
        assert roundtrip(df).equals(df, null_equal=True)

    def test_nan_distinct_from_null(self):
        import polars as pl

        # Polars treats float NaN and null as separate concepts; both must survive the roundtrip.
        s = pl.Series("x", [1.0, float("nan"), None, 4.0])
        result = roundtrip(pl.DataFrame({"x": s}))["x"]
        assert result.is_nan().to_list() == s.is_nan().to_list()
        assert result.is_null().to_list() == s.is_null().to_list()

    def test_all_dtypes_preserved(self):
        import polars as pl

        df = pl.DataFrame(
            {
                "cats": pl.Series(["x", "y", "z"]).cast(pl.Categorical),
                "dates": pl.Series(["2021-01-01", "2021-06-15", "2022-03-10"]).str.to_date(),
                "enums": pl.Series(["a", "b", "a"]).cast(pl.Enum(["a", "b"])),
            }
        )
        result = roundtrip(df)
        assert result.dtypes == df.dtypes
        assert result.equals(df, null_equal=True)

    def test_timezone_aware_datetime(self):
        import polars as pl

        df = pl.DataFrame(
            {
                "ts": pl.Series(["2021-01-01 12:00:00", "2021-06-15 00:00:00"])
                .str.to_datetime()
                .dt.convert_time_zone("Europe/London")
            }
        )
        result = roundtrip(df)
        assert result.dtypes == df.dtypes
        assert result.equals(df, null_equal=True)


class TestPolarsSeries:
    def test_name_and_values_preserved(self):
        import polars as pl

        s = pl.Series("my_col", [1.0, float("nan"), 3.0])
        result = roundtrip(s)
        assert result.name == "my_col"
        assert result.equals(s, null_equal=True)

    def test_categorical_dtype_preserved(self):
        import polars as pl

        s = pl.Series("cat", ["a", "b", "a"]).cast(pl.Categorical)
        result = roundtrip(s)
        assert result.dtype == pl.Categorical
        assert result.equals(s, null_equal=True)


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

    def test_fitted_estimator_with_callable_attribute(self):
        # GradientBoostingRegressor stores _loss as a callable instance; the attribute filter
        # must include callable objects (not just data), or predict() will fail on restore.
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        gbr = GradientBoostingRegressor(n_estimators=10, random_state=42).fit(X, y)
        np.testing.assert_array_equal(roundtrip(gbr).predict(X), gbr.predict(X))


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
