import base64
import datetime
import json
import math
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from functools import singledispatch
from pathlib import Path
from typing import Any

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


def isnamedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_asdict") and callable(obj._asdict)


@singledispatch
def serialize(data: Any) -> Any:
    """Convert a Python object to a JSON-serializable form.

    Uses tagged dict encoding ({"py/type": ...}) for types JSON cannot represent
    natively. Registered handlers cover Python builtins, NumPy, SciPy, pandas,
    and sklearn. Additional types can be supported by registering new handlers
    via @serialize.register(MyType).
    """
    raise TypeError(f"Type {type(data)} not data-serializable")


@serialize.register(type(None))
def serialize_none(data: None) -> None:
    return data


@serialize.register(bool)
def serialize_bool(data: bool) -> bool:
    return data


@serialize.register(int)
def serialize_int(data: int) -> int:
    return data


@serialize.register(str)
def serialize_str(data: str) -> str:
    return data


@serialize.register(list)
def serialize_list(data: list) -> list:
    return [serialize(val) for val in data]


@serialize.register(float)
def serialize_float(data: float) -> float | dict:
    if math.isnan(data):
        return {"py/float": "nan"}
    elif math.isinf(data):
        return {"py/float": "inf" if data > 0 else "-inf"}
    else:
        return data


@serialize.register(OrderedDict)
def serialize_ordereddict(data: OrderedDict) -> dict:
    return {"py/collections.OrderedDict": [[serialize(k), serialize(v)] for k, v in data.items()]}


@serialize.register(tuple)
def serialize_tuple(data: tuple) -> dict:
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
def serialize_dict(data: dict) -> dict:
    if all(isinstance(k, str) for k in data):
        return {k: serialize(v) for k, v in data.items()}
    else:
        return {"py/dict": [[serialize(k), serialize(v)] for k, v in data.items()]}


@serialize.register(set)
def serialize_set(data: set) -> dict:
    return {"py/set": [serialize(val) for val in data]}


@serialize.register(frozenset)
def serialize_frozenset(data: frozenset) -> dict:
    return {"py/frozenset": [serialize(val) for val in data]}


@serialize.register(bytes)
def serialize_bytes(data: bytes) -> dict:
    return {"py/bytes": base64.b64encode(data).decode("ascii")}


@serialize.register(bytearray)
def serialize_bytearray(data: bytearray) -> dict:
    return {"py/bytearray": base64.b64encode(data).decode("ascii")}


@serialize.register(slice)
def serialize_slice(data: slice) -> dict:
    return {"py/slice": {"start": data.start, "stop": data.stop, "step": data.step}}


@serialize.register(datetime.datetime)
def serialize_datetime(data: datetime.datetime) -> dict:
    return {"py/datetime.datetime": data.isoformat()}


@serialize.register(datetime.date)
def serialize_date(data: datetime.date) -> dict:
    return {"py/datetime.date": data.isoformat()}


def restore_python_float(dct: dict) -> float:
    value = dct["py/float"]
    if value == "nan":
        return float("nan")
    elif value == "inf":
        return float("inf")
    else:
        return float("-inf")


def restore_namedtuple(dct: dict) -> tuple:
    data = dct["py/collections.namedtuple"]
    return namedtuple(data["type"], data["fields"])(*data["values"])


def restore_bytes(dct: dict) -> bytes:
    return base64.b64decode(dct["py/bytes"])


def restore_bytearray(dct: dict) -> bytearray:
    return bytearray(base64.b64decode(dct["py/bytearray"]))


def restore_frozenset(dct: dict) -> frozenset:
    return frozenset(dct["py/frozenset"])


def restore_slice(dct: dict) -> slice:
    data = dct["py/slice"]
    return slice(data["start"], data["stop"], data["step"])


def restore_datetime(dct: dict) -> datetime.datetime:
    return datetime.datetime.fromisoformat(dct["py/datetime.datetime"])


def restore_date(dct: dict) -> datetime.date:
    return datetime.date.fromisoformat(dct["py/datetime.date"])


# Populated by the type-specific dispatch modules on import.
RESTORE_FUNCTION_FACTORY: dict[str, Callable[[dict], Any]] = {
    "py/float": restore_python_float,
    "py/dict": lambda dct: {restore(k): restore(v) for k, v in dct["py/dict"]},
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


def restore(dct: Any) -> Any:
    if not isinstance(dct, dict):
        return dct
    for key in RESTORE_FUNCTION_FACTORY:
        if key in dct:
            return RESTORE_FUNCTION_FACTORY[key](dct)
    return dct


def data_to_json(data: Any) -> str:
    """Serialize *data* to a JSON string.

    Supports all Python, NumPy, SciPy, pandas, and sklearn types registered
    with the serialize dispatcher, including fitted estimators and pipelines.
    The result is a self-contained JSON string that can be restored losslessly
    with json_to_data.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler().fit([[1], [2], [3]])
    >>> restored = json_to_data(data_to_json(scaler))
    >>> restored.mean_
    array([2.])
    """
    return json.dumps(serialize(data))


def json_to_data(s: str) -> Any:
    """Restore an object from a JSON string produced by data_to_json.

    Raises ValueError if the JSON references a class from an untrusted module.
    Call trust_module() before deserializing objects from custom packages.
    Only call this on JSON you produced yourself or received from a trusted source.
    """
    return json.loads(s, object_hook=restore)
