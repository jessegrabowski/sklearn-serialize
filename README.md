# sklearn-serialize

JSON serialization for scikit-learn pipelines and the Python/NumPy/SciPy/pandas types that appear inside them. The goal is lossless round-trips: `json_to_data(data_to_json(obj))` reproduces the original object, including fitted model state.

```python
from sklearn_serialize import data_to_json, json_to_data

json_str = data_to_json(fitted_pipeline)
restored  = json_to_data(json_str)

restored.predict(X)  # identical output to the original
```

## Installation

```bash
pip install sklearn-serialize
```

## Supported types

- **sklearn**: `Pipeline`, `FeatureUnion`, `ColumnTransformer`, and any `BaseEstimator` subclass
- **NumPy**: `ndarray`, scalar integer/float types, `datetime64`
- **SciPy**: sparse matrices (`csr`, `csc`, `coo`, `lil`, `dok`)
- **pandas**: `Series`, `DataFrame` (including categorical and datetime columns)
- **Python**: `tuple`, `set`, `frozenset`, `bytes`, `slice`, `OrderedDict`, `namedtuple`, `datetime`

## Custom estimators

Custom estimators work out of the box as long as their class is importable at deserialization time. Call `trust_module` once at startup to allow deserialization from your package:

```python
from sklearn_serialize import trust_module

trust_module("my_package.transformers")
```

The argument is a module prefix — `"my_package"` covers `my_package.transformers`, `my_package.pipelines`, etc. Only exact matches and dotted submodules are allowed; `"my_pack"` does not cover `"my_package"`.

`json_to_data` will raise `ValueError` if it encounters a class from an untrusted module. This prevents arbitrary code execution when deserializing JSON from untrusted sources. Only call `json_to_data` on JSON you produced yourself or received from a trusted source.

The default trusted set covers `sklearn`, `numpy`, `scipy`, `pandas`, `builtins`, and `sklearn_serialize`.

To trust modules globally without calling `trust_module` in every script, create `~/.sklearnserialize`:

```ini
[trusted_modules]
my_package
polars
```

One module prefix per line. Blank lines and lines starting with `#` are ignored. This file is loaded once at import time.
