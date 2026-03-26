import base64
import datetime
import json
from collections import OrderedDict, namedtuple
from functools import singledispatch
from pathlib import Path

# Modules from which classes may be deserialized. Prevents arbitrary code
# execution when loading JSON from untrusted sources. Call trust_module() to
# extend this set for custom estimator packages.
_TRUSTED_MODULES: set[str] = {"builtins", "numpy", "pandas", "scipy", "sklearn", "sklearn_serialize"}


def trust_module(prefix: str) -> None:
    """Allow deserialization of classes whose module starts with *prefix*.

    Call this once at application startup for each package that contains
    custom estimators or transformers you intend to deserialize.

    Example
    -------
    >>> trust_module("mycompany.transformers")
    """
    _TRUSTED_MODULES.add(prefix)


def _check_trusted(module_name: str) -> None:
    if not any(module_name == m or module_name.startswith(m + ".") for m in _TRUSTED_MODULES):
        raise ValueError(
            f"Refusing to deserialize class from untrusted module '{module_name}'. "
            f"Call trust_module('{module_name.split('.')[0]}') to allow it."
        )


def _load_rc_file() -> None:
    rc_path = Path.home() / ".sklearnserialize"
    if not rc_path.exists():
        return
    section = None
    with open(rc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
            elif section == "trusted_modules":
                trust_module(line)


_load_rc_file()


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


@serialize.register(str)
def serialize_str(data):
    return data


@serialize.register(list)
def serialize_list(data):
    return [serialize(val) for val in data]


@serialize.register(float)
def serialize_float(data):
    import math

    if math.isnan(data):
        return {"py/float": "nan"}
    elif math.isinf(data):
        return {"py/float": "inf" if data > 0 else "-inf"}
    else:
        return data


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


@serialize.register(frozenset)
def serialize_frozenset(data):
    return {"py/frozenset": [serialize(val) for val in data]}


@serialize.register(bytes)
def serialize_bytes(data):
    return {"py/bytes": base64.b64encode(data).decode("ascii")}


@serialize.register(bytearray)
def serialize_bytearray(data):
    return {"py/bytearray": base64.b64encode(data).decode("ascii")}


@serialize.register(slice)
def serialize_slice(data):
    return {"py/slice": {"start": data.start, "stop": data.stop, "step": data.step}}


@serialize.register(datetime.datetime)
def serialize_datetime(data):
    return {"py/datetime.datetime": data.isoformat()}


@serialize.register(datetime.date)
def serialize_date(data):
    return {"py/datetime.date": data.isoformat()}


# --- Restore functions ---


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


def restore_bytes(dct):
    return base64.b64decode(dct["py/bytes"])


def restore_bytearray(dct):
    return bytearray(base64.b64decode(dct["py/bytearray"]))


def restore_frozenset(dct):
    return frozenset(dct["py/frozenset"])


def restore_slice(dct):
    data = dct["py/slice"]
    return slice(data["start"], data["stop"], data["step"])


def restore_datetime(dct):
    return datetime.datetime.fromisoformat(dct["py/datetime.datetime"])


def restore_date(dct):
    return datetime.date.fromisoformat(dct["py/datetime.date"])


# Populated by the type-specific dispatch modules on import.
RESTORE_FUNCTION_FACTORY: dict = {
    "py/float": restore_python_float,
    "py/dict": lambda dct: dict(dct["py/dict"]),
    "py/tuple": lambda dct: tuple(dct["py/tuple"]),
    "py/set": lambda dct: set(dct["py/set"]),
    "py/frozenset": restore_frozenset,
    "py/collections.namedtuple": restore_namedtuple,
    "py/collections.OrderedDict": lambda dct: OrderedDict(dct["py/collections.OrderedDict"]),
    "py/bytes": restore_bytes,
    "py/bytearray": restore_bytearray,
    "py/slice": restore_slice,
    "py/datetime.datetime": restore_datetime,
    "py/datetime.date": restore_date,
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
