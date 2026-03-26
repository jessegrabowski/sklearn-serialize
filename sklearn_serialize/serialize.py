import datetime
import json
from collections import OrderedDict, namedtuple
from functools import singledispatch
from io import StringIO

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion, Pipeline


def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_asdict") and callable(obj._asdict)


@singledispatch
def serialize(data):
    raise TypeError(f"Type {type(data)} not data-serializable")


@serialize.register(type(None))
def serialize_none(data):
    return data


@serialize.register(bool)
def serialize_bool(data):
    return data


@serialize.register(int)
def serialize_int(data):
    return data


@serialize.register(float)
def serialize_float(data):
    if np.isnan(data):
        return {"py/float": "nan"}
    elif np.isinf(data):
        return {"py/float": "inf" if data > 0 else "-inf"}
    else:
        return data


@serialize.register(str)
def serialize_str(data):
    return data


@serialize.register(list)
def serialize_list(data):
    return [serialize(val) for val in data]


@serialize.register(OrderedDict)
def serialize_ordereddict(data):
    return {"py/collections.OrderedDict": [[serialize(k), serialize(v)] for k, v in data.items()]}


@serialize.register(tuple)
def serialize_tuple(data):
    if isnamedtuple(data):
        return {
            "py/collections.namedtuple": {
                "type": type(data).__name__,
                "fields": list(data._fields),
                "values": [serialize(getattr(data, f)) for f in data._fields],
            }
        }
    else:
        return {"py/tuple": [serialize(val) for val in data]}


@serialize.register(dict)
def serialize_dict(data):
    if all(isinstance(k, str) for k in data):
        return {k: serialize(v) for k, v in data.items()}
    else:
        return {"py/dict": [[serialize(k), serialize(v)] for k, v in data.items()]}


@serialize.register(set)
def serialize_set(data):
    return {"py/set": [serialize(val) for val in data]}


@serialize.register(np.ndarray)
def serialize_ndarray(data):
    if np.issubdtype(data.dtype, np.datetime64):
        # Convert datetime64 elements to ISO format strings
        values = data.astype(str).tolist()
    elif data.dtype == "object":
        values = [serialize(val) for val in data.tolist()]
    else:
        values = data.tolist()
    return {
        "py/numpy.ndarray": {
            "values": values,
            "dtype": str(data.dtype),
        }
    }


@serialize.register(np.integer)
def serialize_np_integer(data):
    return {"py/numpy.int": int(data)}


@serialize.register(np.floating)
def serialize_np_float(data):
    if np.isnan(data):
        return {"py/numpy.float": "nan"}
    elif np.isinf(data):
        return {"py/numpy.float": "inf" if data > 0 else "-inf"}
    else:
        return {"py/numpy.float": data.item().hex()}


@serialize.register(type)
def serialize_type(data):
    if issubclass(data, np.generic):
        return {"py/numpy.type": data.__name__}
    else:
        return {"py/type": {"module": data.__module__, "qualname": data.__qualname__}}


@serialize.register(bytes)
def serialize_bytes(data):
    return {"py/bytes": data.decode("utf-8", errors="replace")}


@serialize.register(bytearray)
def serialize_bytearray(data):
    return {"py/bytearray": data.decode("utf-8", errors="replace")}


@serialize.register(slice)
def serialize_slice(data):
    return {
        "py/slice": {
            "start": data.start,
            "stop": data.stop,
            "step": data.step,
        }
    }


@serialize.register(frozenset)
def serialize_frozenset(data):
    return {"py/frozenset": [serialize(val) for val in data]}


@serialize.register(datetime.datetime)
def serialize_datetime(data):
    return {"py/datetime.datetime": data.isoformat()}


@serialize.register(datetime.date)
def serialize_date(data):
    return {"py/datetime.date": data.isoformat()}


@serialize.register(np.datetime64)
def serialize_np_datetime64(data):
    return {"py/numpy.datetime64": str(data)}


@serialize.register(sparse.coo_matrix)
def serialize_sparse_coo_matrix(data):
    return {
        "py/scipy.sparse": {
            "format": data.getformat(),
            "data": serialize(data.data),
            "row": serialize(data.row),
            "col": serialize(data.col),
            "shape": data.shape,
        }
    }


@serialize.register(sparse.spmatrix)
def serialize_sparse_matrix(data):
    return {
        "py/scipy.sparse": {
            "data": serialize(data.data),
            "indices": serialize(data.indices),
            "indptr": serialize(data.indptr),
            "shape": data.shape,
            "format": data.getformat(),
        }
    }


@serialize.register(pd.Series)
def serialize_pandas_series(data):
    # Strip the name before calling to_json — pandas embeds it in the JSON string,
    # and complex names (tuples, numpy scalars) cannot survive that round-trip.
    # The name is stored separately and restored via our own serialize/restore.
    json_str = data.rename(None).to_json(orient="split", date_format="iso")
    return {"py/pandas.Series": json_str, "name": serialize(data.name)}


@serialize.register(pd.DataFrame)
def serialize_pandas_dataframe(data):
    json_str = data.to_json(orient="split", date_format="iso")
    categorical_cols = data.select_dtypes(include=["category"]).columns.tolist()
    categorical_attrs = {
        col: {
            "ordered": data[col].cat.ordered,
            "categories": data[col].cat.categories.values.tolist(),
        }
        for col in categorical_cols
    }

    date_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

    return {
        "py/pandas.DataFrame": json_str,
        "categorical_columns": categorical_cols,
        "categorical_attrs": categorical_attrs,
        "date_columns": date_cols,
    }


@serialize.register(BaseEstimator)
def serialize_sklearn_estimator(estimator):
    params = estimator.get_params(deep=False)

    attributes = {
        attr_name: attr_value for attr_name, attr_value in estimator.__dict__.items() if not callable(attr_value)
    }

    data = {
        "class": estimator.__class__.__name__,
        "module": estimator.__class__.__module__,
        "params": serialize(params),
        "attributes": serialize(attributes),
    }
    return {"py/sklearn_estimator": data}


@serialize.register(ColumnTransformer)
def serialize_column_transformer(ct):
    params = ct.get_params(deep=False)
    attributes = {
        attr_name: attr_value
        for attr_name, attr_value in ct.__dict__.items()
        if not callable(attr_value)
        and attr_name not in params
        and (attr_name.endswith("_") or attr_name.startswith("_"))
    }

    data = {
        "class": ct.__class__.__name__,
        "module": ct.__class__.__module__,
        "params": serialize(params),
        "attributes": serialize(attributes),
    }

    return {"py/sklearn.ColumnTransformer": data}


@serialize.register(make_column_selector)
def serialize_make_column_selector(selector):
    data = {
        "class": selector.__class__.__name__,
        "module": selector.__class__.__module__,
        "params": serialize(selector.__dict__),
    }

    return {"py/sklearn.make_column_selector": data}


@serialize.register(FeatureUnion)
def serialize_feature_union(fu):
    transformer_list = []
    for name, transformer in fu.transformer_list:
        transformer_list.append((name, serialize(transformer)))

    params = fu.get_params(deep=False)
    data = {
        "class": fu.__class__.__name__,
        "module": fu.__class__.__module__,
        "params": serialize(params),
        "transformer_list": transformer_list,
    }
    return {"py/sklearn.FeatureUnion": data}


@serialize.register(Pipeline)
def serialize_pipeline(pipeline):
    steps = []
    for name, estimator in pipeline.steps:
        steps.append((name, serialize(estimator)))

    params = pipeline.get_params(deep=False)
    data = {
        "class": pipeline.__class__.__name__,
        "module": pipeline.__class__.__module__,
        "params": serialize(params),
        "steps": steps,
    }
    return {"py/sklearn.Pipeline": data}


def restore_python_float(dct):
    value = dct["py/float"]
    if value == "nan":
        return float("nan")
    elif value == "inf":
        return float("inf")
    elif value == "-inf":
        return float("-inf")
    else:
        return float(value)


def restore_namedtuple(dct):
    data = dct["py/collections.namedtuple"]
    return namedtuple(data["type"], data["fields"])(*data["values"])


def restore_ndarray(dct):
    data = dct["py/numpy.ndarray"]
    values = data["values"]
    dtype_str = data["dtype"]
    if np.issubdtype(np.dtype(dtype_str), np.datetime64):
        restored_values = [np.datetime64(val) for val in values]
    elif dtype_str == "object":
        restored_values = [restore(val) if isinstance(val, dict) else val for val in values]
    else:
        restored_values = values
    return np.array(restored_values, dtype=dtype_str)


def restore_numpy_int(dct):
    return np.int64(dct["py/numpy.int"])


def restore_numpy_float(dct):
    value = dct["py/numpy.float"]
    dtype = np.dtype(dct.get("dtype", "float64"))

    if value == "nan":
        return dtype.type(np.nan)
    elif value == "inf":
        return dtype.type(np.inf)
    elif value == "-inf":
        return dtype.type(-np.inf)
    else:
        return dtype.type(float.fromhex(value))


def restore_numpy_type(dct):
    type_name = dct["py/numpy.type"]
    # Abstract numpy types (e.g. np.number, np.integer) aren't dtype-constructable;
    # fall back to getattr(np, name) which works for all np.generic subclasses.
    numpy_type = getattr(np, type_name, None)
    if numpy_type is not None and isinstance(numpy_type, type) and issubclass(numpy_type, np.generic):
        return numpy_type
    return np.dtype(type_name).type


def restore_type(dct):
    data = dct["py/type"]
    module = __import__(data["module"], fromlist=[data["qualname"]])
    return getattr(module, data["qualname"])


def restore_bytes(dct):
    return dct["py/bytes"].encode("utf-8")


def restore_bytearray(dct):
    return bytearray(dct["py/bytearray"].encode("utf-8"))


def restore_frozenset(dct):
    return frozenset(dct["py/frozenset"])


def restore_slice(dct):
    data = dct["py/slice"]
    return slice(data["start"], data["stop"], data["step"])


def restore_datetime(dct):
    return datetime.datetime.fromisoformat(dct["py/datetime.datetime"])


def restore_date(dct):
    return datetime.date.fromisoformat(dct["py/datetime.date"])


def restore_np_datetime64(dct):
    return np.datetime64(dct["py/numpy.datetime64"])


def restore_sparse_matrix(dct):
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, dok_matrix, lil_matrix

    sparse_format_factory = {
        "csr": csr_matrix,
        "csc": csc_matrix,
        "coo": coo_matrix,
        "lil": lil_matrix,
        "dok": dok_matrix,
    }
    data = dct["py/scipy.sparse"]
    format = data["format"]
    shape = tuple(data["shape"])
    data_array = restore(data["data"])

    constructor = sparse_format_factory.get(format)

    if format in ["csr", "csc"]:
        indices = restore(data["indices"])
        indptr = restore(data["indptr"])

        matrix = constructor((data_array, indices, indptr), shape=shape)
    elif format == "coo":
        row = restore(data["row"])
        col = restore(data["col"])
        matrix = constructor((data_array, (row, col)), shape=shape)
    else:
        matrix = constructor(restore(data["dense"]), shape=shape)

    return matrix


def restore_pandas_series(dct):
    json_str = dct["py/pandas.Series"]
    series = pd.read_json(StringIO(json_str), typ="series", orient="split")
    series.name = dct.get("name", None)
    return series


def restore_pandas_dataframe(dct):
    json_str = dct["py/pandas.DataFrame"]
    df = pd.read_json(StringIO(json_str), orient="split", convert_dates=dct.get("date_columns", []))
    categorical_cols = dct.get("categorical_columns", [])
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col], **dct["categorical_attrs"][col])
    return df


def restore_sklearn_estimator(dct):
    data = dct["py/sklearn_estimator"]
    class_name = data["class"]
    module_name = data["module"]
    params = restore(data["params"])
    attributes = restore(data["attributes"])

    module = __import__(module_name, fromlist=[class_name])
    estimator_class = getattr(module, class_name)

    estimator = estimator_class(**params)

    for attr, value in attributes.items():
        setattr(estimator, attr, value)

    return estimator


def restore_column_transformer(dct):
    data = dct["py/sklearn.ColumnTransformer"]
    class_name = data["class"]
    module_name = data["module"]
    params = restore(data["params"])
    attributes = restore(data["attributes"])

    module = __import__(module_name, fromlist=[class_name])
    ct_class = getattr(module, class_name)

    ct = ct_class(**params)

    for attribute, value in attributes.items():
        setattr(ct, attribute, value)

    return ct


def restore_make_column_selector(dct):
    data = dct["py/sklearn.make_column_selector"]
    class_name = data["class"]
    module_name = data["module"]
    params = restore(data["params"])

    module = __import__(module_name, fromlist=[class_name])
    selector_class = getattr(module, class_name)

    selector = selector_class(**params)
    return selector


def restore_feature_union(dct):
    data = dct["py/sklearn.FeatureUnion"]
    class_name = data["class"]
    module_name = data["module"]
    params = restore(data["params"])

    module = __import__(module_name, fromlist=[class_name])
    fu_class = getattr(module, class_name)

    fu = fu_class(**params)
    return fu


def restore_pipeline(dct):
    data = dct["py/sklearn.Pipeline"]
    class_name = data["class"]
    module_name = data["module"]
    params = restore(data["params"])

    module = __import__(module_name, fromlist=[class_name])
    pipeline_class = getattr(module, class_name)

    pipeline = pipeline_class(**params)
    return pipeline


RESTORE_FUNCTION_FACTORY = {
    "py/dict": lambda dct: dict(dct["py/dict"]),
    "py/tuple": lambda dct: tuple(dct["py/tuple"]),
    "py/set": lambda dct: set(dct["py/set"]),
    "py/frozenset": restore_frozenset,
    "py/collections.namedtuple": restore_namedtuple,
    "py/numpy.ndarray": restore_ndarray,
    "py/collections.OrderedDict": lambda dct: OrderedDict(dct["py/collections.OrderedDict"]),
    "py/bytes": restore_bytes,
    "py/bytearray": restore_bytearray,
    "py/numpy.int": restore_numpy_int,
    "py/numpy.float": restore_numpy_float,
    "py/numpy.type": restore_numpy_type,
    "py/type": restore_type,
    "py/float": restore_python_float,
    "py/datetime.datetime": restore_datetime,
    "py/datetime.date": restore_date,
    "py/numpy.datetime64": restore_np_datetime64,
    "py/scipy.sparse": restore_sparse_matrix,
    "py/sklearn_estimator": restore_sklearn_estimator,
    "py/sklearn.ColumnTransformer": restore_column_transformer,
    "py/sklearn.FeatureUnion": restore_feature_union,
    "py/sklearn.Pipeline": restore_pipeline,
    "py/pandas.DataFrame": restore_pandas_dataframe,
    "py/pandas.Series": restore_pandas_series,
    "py/slice": restore_slice,
    "py/sklearn.make_column_selector": restore_make_column_selector,
}


def restore(dct):
    for key in RESTORE_FUNCTION_FACTORY:
        if key in dct:
            return RESTORE_FUNCTION_FACTORY[key](dct)
    return dct


def data_to_json(data) -> str:
    return json.dumps(serialize(data))


def json_to_data(s: str):
    return json.loads(s, object_hook=restore)
