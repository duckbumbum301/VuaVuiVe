from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM footprint.

    Notes:
    - Keeps object/string columns unchanged.
    - Uses signed ints only (safe for Instacart-like IDs).
    """

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory usage ban đầu: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
            continue

        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
            continue

        if pd.api.types.is_float_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
            continue

        # For other numeric-like types (e.g. boolean), keep as is.

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory usage sau khi tối ưu: {end_mem:.2f} MB")
        if start_mem > 0:
            print(f"Đã giảm: {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def read_csv_optimized(
    csv_path: Path | str,
    *,
    chunksize: Optional[int] = None,
    reduce_memory: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read CSV with optional chunking and memory downcasting.

    Why chunking matters:
    - For very large files (e.g. order_products__prior.csv), reading in chunks
      avoids a huge peak RAM usage before downcasting.

    Parameters
    - chunksize: number of rows per chunk. If None, reads in one shot.
    - reduce_memory: apply reduce_mem_usage to the resulting DataFrame.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if chunksize is None:
        df = pd.read_csv(csv_path, **read_csv_kwargs)
        return reduce_mem_usage(df) if reduce_memory else df

    chunks = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, **read_csv_kwargs), start=1):
        if reduce_memory:
            chunk = reduce_mem_usage(chunk, verbose=(i == 1))
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return df


def safe_to_parquet(df: pd.DataFrame, output_path: Path | str, *, index: bool = False) -> None:
    """Write parquet if possible; otherwise print a helpful message."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(output_path, index=index)
        print(f"Saved: {output_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Không thể ghi Parquet ({output_path.name}). Lý do: {exc}")
        print("Gợi ý: cài 'pyarrow' (khuyến nghị) rồi chạy lại: pip install pyarrow")
