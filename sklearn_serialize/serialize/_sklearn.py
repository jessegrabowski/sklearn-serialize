import inspect
from typing import Any

import numpy as np
from sklearn._loss.loss import BaseLoss
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree._tree import Tree

from ._core import RESTORE_FUNCTION_FACTORY, _check_trusted, restore, serialize


@serialize.register(BaseLoss)
def serialize_loss(data: BaseLoss) -> dict:
    cls = type(data)
    sig = inspect.signature(cls.__init__)
    params = {}
    for p in sig.parameters:
        if p in ("self", "sample_weight"):
            continue
        # Parameters may live on self or inside the Cython sub-object (closs).
        v = getattr(data, p, None)
        if v is None:
            v = getattr(getattr(data, "closs", None), p, None)
        if v is not None:
            params[p] = v
    return {"py/sklearn.BaseLoss": {"class": cls.__name__, "module": cls.__module__, "params": params}}


@serialize.register(BaseEstimator)
def serialize_sklearn_estimator(estimator: BaseEstimator) -> dict:
    params = estimator.get_params(deep=False)
    attributes = {k: v for k, v in estimator.__dict__.items() if not (inspect.isfunction(v) or inspect.ismethod(v))}
    return {
        "py/sklearn_estimator": {
            "class": estimator.__class__.__name__,
            "module": estimator.__class__.__module__,
            "params": serialize(params),
            "attributes": serialize(attributes),
        }
    }


@serialize.register(ColumnTransformer)
def serialize_column_transformer(ct: ColumnTransformer) -> dict:
    params = ct.get_params(deep=False)
    attributes = {
        k: v
        for k, v in ct.__dict__.items()
        if not callable(v) and k not in params and (k.endswith("_") or k.startswith("_"))
    }
    return {
        "py/sklearn.ColumnTransformer": {
            "class": ct.__class__.__name__,
            "module": ct.__class__.__module__,
            "params": serialize(params),
            "attributes": serialize(attributes),
        }
    }


@serialize.register(make_column_selector)
def serialize_make_column_selector(selector: make_column_selector) -> dict:
    return {
        "py/sklearn.make_column_selector": {
            "class": selector.__class__.__name__,
            "module": selector.__class__.__module__,
            "params": serialize(selector.__dict__),
        }
    }


@serialize.register(FeatureUnion)
def serialize_feature_union(fu: FeatureUnion) -> dict:
    params = fu.get_params(deep=False)
    return {
        "py/sklearn.FeatureUnion": {
            "class": fu.__class__.__name__,
            "module": fu.__class__.__module__,
            "params": serialize(params),
        }
    }


@serialize.register(Pipeline)
def serialize_pipeline(pipeline: Pipeline) -> dict:
    params = pipeline.get_params(deep=False)
    return {
        "py/sklearn.Pipeline": {
            "class": pipeline.__class__.__name__,
            "module": pipeline.__class__.__module__,
            "params": serialize(params),
        }
    }


@serialize.register(Tree)
def serialize_tree(data: Tree) -> dict:
    state = data.__getstate__()
    nodes = state["nodes"]
    # nodes is a structured array whose str(dtype) is not parseable by np.dtype();
    # serialize each named column as a plain array and reconstruct the dtype using
    # the full dict descriptor (names/formats/offsets/itemsize) to preserve padding.
    nodes_dtype = {
        "names": list(nodes.dtype.names),
        "formats": [str(nodes.dtype.fields[name][0]) for name in nodes.dtype.names],
        "offsets": [nodes.dtype.fields[name][1] for name in nodes.dtype.names],
        "itemsize": nodes.dtype.itemsize,
    }
    return {
        "py/sklearn.Tree": {
            "max_depth": state["max_depth"],
            "node_count": state["node_count"],
            "nodes_fields": {name: serialize(nodes[name]) for name in nodes.dtype.names},
            "nodes_dtype": nodes_dtype,
            "values": serialize(state["values"]),
            "n_features": data.n_features,
            "n_classes": serialize(data.n_classes),
            "n_outputs": data.n_outputs,
        }
    }


def restore_loss(dct: dict) -> BaseLoss:
    data = dct["py/sklearn.BaseLoss"]
    cls = _import_class(data["module"], data["class"])
    return cls(**data["params"])


def _import_class(module_name: str, class_name: str) -> type:
    _check_trusted(module_name)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def restore_sklearn_estimator(dct: dict) -> BaseEstimator:
    data = dct["py/sklearn_estimator"]
    cls = _import_class(data["module"], data["class"])
    params = restore(data["params"])
    attributes = restore(data["attributes"])
    estimator = cls(**params)
    for attr, value in attributes.items():
        setattr(estimator, attr, value)
    return estimator


def restore_column_transformer(dct: dict) -> ColumnTransformer:
    data = dct["py/sklearn.ColumnTransformer"]
    cls = _import_class(data["module"], data["class"])
    params = restore(data["params"])
    attributes = restore(data["attributes"])
    ct = cls(**params)
    for attr, value in attributes.items():
        setattr(ct, attr, value)
    return ct


def restore_make_column_selector(dct: dict) -> Any:
    data = dct["py/sklearn.make_column_selector"]
    cls = _import_class(data["module"], data["class"])
    return cls(**restore(data["params"]))


def restore_feature_union(dct: dict) -> FeatureUnion:
    data = dct["py/sklearn.FeatureUnion"]
    cls = _import_class(data["module"], data["class"])
    return cls(**restore(data["params"]))


def restore_pipeline(dct: dict) -> Pipeline:
    data = dct["py/sklearn.Pipeline"]
    cls = _import_class(data["module"], data["class"])
    return cls(**restore(data["params"]))


def restore_tree(dct: dict) -> Tree:
    data = dct["py/sklearn.Tree"]
    dt = data["nodes_dtype"]
    nodes_dtype = np.dtype(
        {"names": dt["names"], "formats": dt["formats"], "offsets": dt["offsets"], "itemsize": dt["itemsize"]}
    )
    nodes = np.zeros(data["node_count"], dtype=nodes_dtype)
    for name, col in data["nodes_fields"].items():
        nodes[name] = col
    state = {
        "max_depth": data["max_depth"],
        "node_count": data["node_count"],
        "nodes": nodes,
        "values": data["values"],
    }
    n_classes = data["n_classes"]
    tree = Tree(data["n_features"], n_classes, data["n_outputs"])
    tree.__setstate__(state)
    return tree


RESTORE_FUNCTION_FACTORY.update(
    {
        "py/sklearn.BaseLoss": restore_loss,
        "py/sklearn.Tree": restore_tree,
        "py/sklearn_estimator": restore_sklearn_estimator,
        "py/sklearn.ColumnTransformer": restore_column_transformer,
        "py/sklearn.make_column_selector": restore_make_column_selector,
        "py/sklearn.FeatureUnion": restore_feature_union,
        "py/sklearn.Pipeline": restore_pipeline,
    }
)
