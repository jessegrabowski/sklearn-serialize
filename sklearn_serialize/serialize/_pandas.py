from io import StringIO

import pandas as pd

from ._core import RESTORE_FUNCTION_FACTORY, serialize


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


RESTORE_FUNCTION_FACTORY.update(
    {
        "py/pandas.Series": restore_pandas_series,
        "py/pandas.DataFrame": restore_pandas_dataframe,
    }
)
