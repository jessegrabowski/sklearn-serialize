import numpy as np

from ._core import RESTORE_FUNCTION_FACTORY, _check_trusted, restore, serialize


@serialize.register(np.ndarray)
def serialize_ndarray(data):
    if np.issubdtype(data.dtype, np.datetime64):
        # Convert datetime64 elements to ISO format strings
        values = data.astype(str).tolist()
    elif data.dtype == "object":
        values = [serialize(val) for val in data.tolist()]
    else:
        values = data.tolist()
    return {"py/numpy.ndarray": {"values": values, "dtype": str(data.dtype)}}


@serialize.register(np.integer)
def serialize_np_integer(data):
    return {"py/numpy.int": int(data), "dtype": type(data).__name__}


@serialize.register(np.floating)
def serialize_np_float(data):
    if np.isnan(data):
        return {"py/numpy.float": "nan", "dtype": type(data).__name__}
    elif np.isinf(data):
        return {"py/numpy.float": "inf" if data > 0 else "-inf", "dtype": type(data).__name__}
    else:
        return {"py/numpy.float": data.item().hex(), "dtype": type(data).__name__}


@serialize.register(type)
def serialize_type(data):
    if issubclass(data, np.generic):
        return {"py/numpy.type": data.__name__}
    else:
        return {"py/type": {"module": data.__module__, "qualname": data.__qualname__}}


@serialize.register(np.datetime64)
def serialize_np_datetime64(data):
    return {"py/numpy.datetime64": str(data)}


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
    dtype = getattr(np, dct.get("dtype", "int64"))
    return dtype(dct["py/numpy.int"])


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
    _check_trusted(data["module"])
    module = __import__(data["module"], fromlist=[data["qualname"]])
    return getattr(module, data["qualname"])


def restore_np_datetime64(dct):
    return np.datetime64(dct["py/numpy.datetime64"])


RESTORE_FUNCTION_FACTORY.update(
    {
        "py/numpy.ndarray": restore_ndarray,
        "py/numpy.int": restore_numpy_int,
        "py/numpy.float": restore_numpy_float,
        "py/numpy.type": restore_numpy_type,
        "py/type": restore_type,
        "py/numpy.datetime64": restore_np_datetime64,
    }
)
