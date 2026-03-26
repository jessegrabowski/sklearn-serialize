from typing import Any

import numpy as np
from numpy.random import RandomState

from ._core import RESTORE_FUNCTION_FACTORY, _check_trusted, restore, serialize


@serialize.register(np.ndarray)
def serialize_ndarray(data: np.ndarray) -> dict:
    if np.issubdtype(data.dtype, np.datetime64):
        # datetime64 has no JSON representation; ISO strings round-trip losslessly
        values = data.astype(str).tolist()
    elif np.issubdtype(data.dtype, np.complexfloating):
        # JSON has no complex type; view as interleaved real pairs and view back on restore
        values = data.view(data.real.dtype).tolist()
    elif data.dtype == "object":
        values = [serialize(val) for val in data.tolist()]
    else:
        values = data.tolist()
    return {"py/numpy.ndarray": {"values": values, "dtype": str(data.dtype)}}


@serialize.register(np.integer)
def serialize_np_integer(data: np.integer) -> dict:
    return {"py/numpy.int": int(data), "dtype": type(data).__name__}


@serialize.register(np.floating)
def serialize_np_float(data: np.floating) -> dict:
    if np.isnan(data):
        return {"py/numpy.float": "nan", "dtype": type(data).__name__}
    elif np.isinf(data):
        return {"py/numpy.float": "inf" if data > 0 else "-inf", "dtype": type(data).__name__}
    else:
        return {"py/numpy.float": data.item().hex(), "dtype": type(data).__name__}


@serialize.register(np.complexfloating)
def serialize_np_complex(data: np.complexfloating) -> dict:
    def _encode(v: np.floating) -> str:
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v.item().hex()

    return {"py/numpy.complex": {"real": _encode(data.real), "imag": _encode(data.imag), "dtype": type(data).__name__}}


@serialize.register(type)
def serialize_type(data: type) -> dict:
    if issubclass(data, np.generic):
        return {"py/numpy.type": data.__name__}
    else:
        return {"py/type": {"module": data.__module__, "qualname": data.__qualname__}}


@serialize.register(np.datetime64)
def serialize_np_datetime64(data: np.datetime64) -> dict:
    return {"py/numpy.datetime64": str(data)}


def restore_ndarray(dct: dict) -> np.ndarray:
    data = dct["py/numpy.ndarray"]
    values = data["values"]
    dtype_str = data["dtype"]
    if np.issubdtype(np.dtype(dtype_str), np.datetime64):
        restored_values = [np.datetime64(val) for val in values]
    elif np.issubdtype(np.dtype(dtype_str), np.complexfloating):
        real_dtype = np.empty(0, dtype=dtype_str).real.dtype
        return np.array(values, dtype=real_dtype).view(np.dtype(dtype_str))
    elif dtype_str == "object":
        restored_values = [restore(val) if isinstance(val, dict) else val for val in values]
    else:
        restored_values = values
    return np.array(restored_values, dtype=dtype_str)


def restore_numpy_int(dct: dict) -> np.integer:
    dtype = getattr(np, dct.get("dtype", "int64"))
    return dtype(dct["py/numpy.int"])


def restore_numpy_float(dct: dict) -> np.floating:
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


def restore_numpy_complex(dct: dict) -> np.complexfloating:
    data = dct["py/numpy.complex"]
    dtype = np.dtype(data["dtype"])

    def _decode(s: str) -> float:
        if s == "nan":
            return float("nan")
        if s == "inf":
            return float("inf")
        if s == "-inf":
            return float("-inf")
        return float.fromhex(s)

    return dtype.type(complex(_decode(data["real"]), _decode(data["imag"])))


def restore_numpy_type(dct: dict) -> type:
    type_name = dct["py/numpy.type"]
    # Abstract numpy types (e.g. np.number, np.integer) aren't dtype-constructable;
    # fall back to getattr(np, name) which works for all np.generic subclasses.
    numpy_type = getattr(np, type_name, None)
    if numpy_type is not None and isinstance(numpy_type, type) and issubclass(numpy_type, np.generic):
        return numpy_type
    return np.dtype(type_name).type


def restore_type(dct: dict) -> Any:
    data = dct["py/type"]
    _check_trusted(data["module"])
    module = __import__(data["module"], fromlist=[data["qualname"]])
    return getattr(module, data["qualname"])


def restore_np_datetime64(dct: dict) -> np.datetime64:
    return np.datetime64(dct["py/numpy.datetime64"])


@serialize.register(RandomState)
def serialize_random_state(data: RandomState) -> dict:
    return {"py/numpy.RandomState": serialize(data.__getstate__())}


def restore_random_state(dct: dict) -> RandomState:
    rs = RandomState()
    rs.__setstate__(dct["py/numpy.RandomState"])
    return rs


RESTORE_FUNCTION_FACTORY.update(
    {
        "py/numpy.ndarray": restore_ndarray,
        "py/numpy.int": restore_numpy_int,
        "py/numpy.float": restore_numpy_float,
        "py/numpy.complex": restore_numpy_complex,
        "py/numpy.type": restore_numpy_type,
        "py/type": restore_type,
        "py/numpy.datetime64": restore_np_datetime64,
        "py/numpy.RandomState": restore_random_state,
    }
)
