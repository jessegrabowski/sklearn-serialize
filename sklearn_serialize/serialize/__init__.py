# Side-effect imports: register all type-specific dispatch handlers and
# populate RESTORE_FUNCTION_FACTORY. Order matters for singledispatch MRO
# (scipy subclasses before spmatrix base; sklearn subclasses before BaseEstimator).
from . import _numpy, _pandas, _polars, _scipy, _sklearn  # noqa: F401
from ._core import RESTORE_FUNCTION_FACTORY, data_to_json, json_to_data, restore, serialize, trust_module

__all__ = ["data_to_json", "json_to_data", "restore", "serialize", "trust_module", "RESTORE_FUNCTION_FACTORY"]
