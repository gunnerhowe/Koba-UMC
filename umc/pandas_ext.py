"""Pandas/Arrow integration for UMC compression.

Usage:
    import umc.pandas_ext  # registers the accessor

    # Compress a DataFrame
    compressed = df.umc.compress(mode="lossless")
    recovered = umc.pandas_ext.decompress_dataframe(compressed)

    # Or use the functional API
    compressed = umc.pandas_ext.compress_dataframe(df, mode="near_lossless")
    recovered = umc.pandas_ext.decompress_dataframe(compressed)
"""

import struct

import numpy as np
import pandas as pd

import umc


# ---- Functional API ----

def compress_dataframe(
    df: pd.DataFrame,
    mode: str = "lossless",
    columns: list = None,
) -> bytes:
    """Compress a Pandas DataFrame to UMC bytes.

    Preserves column names, index, and dtypes for lossless modes.

    Args:
        df: DataFrame to compress.
        mode: UMC compression mode.
        columns: Specific columns to compress (default: all numeric columns).

    Returns:
        Compressed bytes with UMCD magic header.
    """
    if columns is not None:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame")

    data = numeric_df.values.astype(np.float32)

    # Serialize metadata: column names + original dtypes
    col_names = list(numeric_df.columns)
    meta = {
        "columns": col_names,
        "dtypes": [str(numeric_df[c].dtype) for c in col_names],
        "n_rows": len(df),
    }

    # Check if index should be preserved
    if not isinstance(df.index, pd.RangeIndex):
        meta["index_name"] = df.index.name
        if isinstance(df.index, pd.DatetimeIndex):
            meta["index_type"] = "datetime"
            meta["index_values"] = df.index.astype(np.int64).tolist()
        else:
            meta["index_type"] = "generic"
            meta["index_values"] = [str(v) for v in df.index]

    import json
    meta_bytes = json.dumps(meta).encode("utf-8")

    compressed = umc.compress(data, mode=mode)

    # Format: UMCD + meta_size(4) + meta + compressed
    magic = b"UMCD"
    header = struct.pack("<I", len(meta_bytes))
    return magic + header + meta_bytes + compressed


def decompress_dataframe(data: bytes) -> pd.DataFrame:
    """Decompress UMC bytes back to a Pandas DataFrame.

    Args:
        data: Compressed bytes from compress_dataframe().

    Returns:
        Pandas DataFrame with original column names and index.
    """
    import json

    if len(data) < 4 or data[:4] != b"UMCD":
        raise ValueError("Not a UMC DataFrame stream (missing UMCD magic)")

    meta_size = struct.unpack("<I", data[4:8])[0]
    meta = json.loads(data[8:8 + meta_size].decode("utf-8"))
    compressed = data[8 + meta_size:]

    arr = umc.decompress(compressed)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])

    df = pd.DataFrame(arr, columns=meta["columns"])

    # Restore dtypes
    for col, dtype_str in zip(meta["columns"], meta["dtypes"]):
        try:
            df[col] = df[col].astype(dtype_str)
        except (ValueError, TypeError):
            pass  # keep float32 if conversion fails

    # Restore index
    if "index_type" in meta:
        if meta["index_type"] == "datetime":
            idx = pd.DatetimeIndex(
                np.array(meta["index_values"], dtype=np.int64),
                name=meta.get("index_name"),
            )
            df.index = idx
        elif meta["index_type"] == "generic":
            df.index = pd.Index(meta["index_values"], name=meta.get("index_name"))

    return df


# ---- Arrow support ----

def compress_table(table, mode: str = "lossless") -> bytes:
    """Compress a PyArrow Table to UMC bytes.

    Args:
        table: PyArrow Table.
        mode: UMC compression mode.

    Returns:
        Compressed bytes.
    """
    df = table.to_pandas()
    return compress_dataframe(df, mode=mode)


def decompress_table(data: bytes):
    """Decompress UMC bytes back to a PyArrow Table.

    Args:
        data: Compressed bytes from compress_table().

    Returns:
        PyArrow Table.
    """
    import pyarrow as pa
    df = decompress_dataframe(data)
    return pa.Table.from_pandas(df)


# ---- Pandas accessor ----

@pd.api.extensions.register_dataframe_accessor("umc")
class UMCAccessor:
    """Pandas DataFrame accessor for UMC compression.

    Usage:
        import umc.pandas_ext

        compressed = df.umc.compress(mode="lossless")
        ratio = df.umc.estimate_ratio()
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def compress(self, mode: str = "lossless", columns: list = None) -> bytes:
        """Compress this DataFrame to UMC bytes.

        Args:
            mode: UMC compression mode.
            columns: Specific columns (default: all numeric).

        Returns:
            Compressed bytes.
        """
        return compress_dataframe(self._obj, mode=mode, columns=columns)

    def estimate_ratio(self, mode: str = "lossless") -> float:
        """Estimate compression ratio without returning compressed data.

        Returns:
            Compression ratio (original_size / compressed_size).
        """
        compressed = self.compress(mode=mode)
        numeric = self._obj.select_dtypes(include=[np.number])
        raw = numeric.values.nbytes
        return raw / max(len(compressed), 1)
