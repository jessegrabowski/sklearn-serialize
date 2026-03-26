from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion, Pipeline

from ._core import RESTORE_FUNCTION_FACTORY, _check_trusted, restore, serialize


@serialize.register(BaseEstimator)
def serialize_sklearn_estimator(estimator):
    params = estimator.get_params(deep=False)
    attributes = {k: v for k, v in estimator.__dict__.items() if not callable(v)}
    return {
        "py/sklearn_estimator": {
            "class": estimator.__class__.__name__,
            "module": estimator.__class__.__module__,
            "params": serialize(params),
            "attributes": serialize(attributes),
        }
    }


@serialize.register(ColumnTransformer)
def serialize_column_transformer(ct):
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
def serialize_make_column_selector(selector):
    return {
        "py/sklearn.make_column_selector": {
            "class": selector.__class__.__name__,
            "module": selector.__class__.__module__,
            "params": serialize(selector.__dict__),
        }
    }


@serialize.register(FeatureUnion)
def serialize_feature_union(fu):
    params = fu.get_params(deep=False)
    return {
        "py/sklearn.FeatureUnion": {
            "class": fu.__class__.__name__,
            "module": fu.__class__.__module__,
            "params": serialize(params),
        }
    }


@serialize.register(Pipeline)
def serialize_pipeline(pipeline):
    params = pipeline.get_params(deep=False)
    return {
        "py/sklearn.Pipeline": {
            "class": pipeline.__class__.__name__,
            "module": pipeline.__class__.__module__,
            "params": serialize(params),
        }
    }


def _import_class(module_name, class_name):
    _check_trusted(module_name)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def restore_sklearn_estimator(dct):
    data = dct["py/sklearn_estimator"]
    cls = _import_class(data["module"], data["class"])
    params = restore(data["params"])
    attributes = restore(data["attributes"])
    estimator = cls(**params)
    for attr, value in attributes.items():
        setattr(estimator, attr, value)
    return estimator


def restore_column_transformer(dct):
    data = dct["py/sklearn.ColumnTransformer"]
    cls = _import_class(data["module"], data["class"])
    params = restore(data["params"])
    attributes = restore(data["attributes"])
    ct = cls(**params)
    for attr, value in attributes.items():
        setattr(ct, attr, value)
    return ct


def restore_make_column_selector(dct):
    data = dct["py/sklearn.make_column_selector"]
    cls = _import_class(data["module"], data["class"])
    return cls(**restore(data["params"]))


def restore_feature_union(dct):
    data = dct["py/sklearn.FeatureUnion"]
    cls = _import_class(data["module"], data["class"])
    return cls(**restore(data["params"]))


def restore_pipeline(dct):
    data = dct["py/sklearn.Pipeline"]
    cls = _import_class(data["module"], data["class"])
    return cls(**restore(data["params"]))


RESTORE_FUNCTION_FACTORY.update(
    {
        "py/sklearn_estimator": restore_sklearn_estimator,
        "py/sklearn.ColumnTransformer": restore_column_transformer,
        "py/sklearn.make_column_selector": restore_make_column_selector,
        "py/sklearn.FeatureUnion": restore_feature_union,
        "py/sklearn.Pipeline": restore_pipeline,
    }
)
