import base64
import io

from ._core import RESTORE_FUNCTION_FACTORY, serialize

try:
    import polars as pl
except ImportError:
    pass
else:

    @serialize.register(pl.DataFrame)
    def serialize_polars_dataframe(data: pl.DataFrame) -> dict:
        buf = io.BytesIO()
        data.write_ipc(buf)
        return {"py/polars.DataFrame": base64.b64encode(buf.getvalue()).decode("ascii")}

    @serialize.register(pl.Series)
    def serialize_polars_series(data: pl.Series) -> dict:
        # Wrap in a DataFrame to carry name and dtype through the IPC round-trip.
        buf = io.BytesIO()
        data.to_frame().write_ipc(buf)
        return {"py/polars.Series": base64.b64encode(buf.getvalue()).decode("ascii")}

    def restore_polars_dataframe(dct: dict) -> pl.DataFrame:
        return pl.read_ipc(io.BytesIO(base64.b64decode(dct["py/polars.DataFrame"])))

    def restore_polars_series(dct: dict) -> pl.Series:
        df = pl.read_ipc(io.BytesIO(base64.b64decode(dct["py/polars.Series"])))
        return df.get_column(df.columns[0])

    RESTORE_FUNCTION_FACTORY.update(
        {
            "py/polars.DataFrame": restore_polars_dataframe,
            "py/polars.Series": restore_polars_series,
        }
    )
