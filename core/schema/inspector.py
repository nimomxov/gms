"""
GMS — CSVInspector  v2.3

Inspects raw CSV or in-memory DataFrame headers and basic statistics.
Does NOT assign semantic meaning — that is SemanticFieldMapper's job.

Outputs a FieldInventory: list of FieldInfo objects describing what
is present in the data before any semantic interpretation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("gms.schema.inspector")


@dataclass
class FieldInfo:
    """Description of a single discovered CSV field."""
    name: str                  # original column name (lowercased)
    dtype: str                 # pandas dtype string
    n_valid: int               # non-null count
    n_total: int               # total row count
    missing_fraction: float    # fraction of nulls
    vmin: Optional[float]      # numeric min (None for non-numeric)
    vmax: Optional[float]      # numeric max
    vmean: Optional[float]     # numeric mean
    vstd: Optional[float]      # numeric std
    is_monotonic: bool         # True if values are monotonically increasing (timestamp-like)
    is_binary: bool            # True if only 0/1 values found
    is_likely_index: bool      # True if it looks like a row index (0..N-1)


@dataclass
class FieldInventory:
    """Complete introspection result for one CSV/stream."""
    source: str                    # filename or 'stream'
    n_rows: int
    n_fields: int
    fields: list[FieldInfo] = field(default_factory=list)
    raw_columns: list[str] = field(default_factory=list)    # original names
    warnings: list[str] = field(default_factory=list)

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def get(self, name: str) -> Optional[FieldInfo]:
        for f in self.fields:
            if f.name == name:
                return f
        return None


class CSVInspector:
    """
    Stage 0 of the telemetry pipeline: raw field discovery.

    Does NOT interpret the data — just inventories what is physically
    present so the SemanticFieldMapper can make informed decisions.

    Usage:
        inspector = CSVInspector()
        inventory = inspector.inspect_file("scan_A.csv")
        # or:
        inventory = inspector.inspect_dataframe(df, source="ble_stream")
    """

    def inspect_file(self, filepath: str | Path) -> FieldInventory:
        """Load and inspect a CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSVInspector: file not found: {path}")

        try:
            df = pd.read_csv(path, comment="#")
        except Exception as e:
            raise ValueError(f"CSVInspector: parse error in {path.name}: {e}")

        return self.inspect_dataframe(df, source=path.name)

    def inspect_dataframe(self, df: pd.DataFrame, source: str = "dataframe") -> FieldInventory:
        """Inspect an already-loaded DataFrame."""
        warnings: list[str] = []
        raw_columns = list(df.columns)

        # Normalize column names
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        n_rows = len(df)
        fields: list[FieldInfo] = []

        for col in df.columns:
            series = df[col]
            n_total = n_rows
            n_valid = int(series.notna().sum())
            missing_frac = (n_total - n_valid) / max(n_total, 1)

            if missing_frac > 0.5:
                warnings.append(f"Field '{col}': {missing_frac:.0%} missing — unreliable")

            numeric = pd.to_numeric(series, errors="coerce")
            is_numeric = numeric.notna().sum() > 0.8 * n_valid

            if is_numeric:
                vals = numeric.dropna().values
                vmin = float(vals.min()) if len(vals) > 0 else None
                vmax = float(vals.max()) if len(vals) > 0 else None
                vmean = float(vals.mean()) if len(vals) > 0 else None
                vstd = float(vals.std()) if len(vals) > 0 else None
                is_monotonic = bool(np.all(np.diff(vals) >= 0)) if len(vals) > 1 else False
                unique_vals = set(vals.tolist())
                is_binary = unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}
                is_likely_index = (
                    is_monotonic
                    and vmin == 0.0
                    and abs(vmax - (n_valid - 1)) < 1e-3
                )
            else:
                vmin = vmax = vmean = vstd = None
                is_monotonic = False
                is_binary = False
                is_likely_index = False

            fields.append(FieldInfo(
                name=col,
                dtype=str(series.dtype),
                n_valid=n_valid,
                n_total=n_total,
                missing_fraction=missing_frac,
                vmin=vmin,
                vmax=vmax,
                vmean=vmean,
                vstd=vstd,
                is_monotonic=is_monotonic,
                is_binary=is_binary,
                is_likely_index=is_likely_index,
            ))

        logger.debug(
            f"[CSVInspector] {source}: {n_rows} rows, "
            f"{len(fields)} fields: {[f.name for f in fields]}"
        )

        return FieldInventory(
            source=source,
            n_rows=n_rows,
            n_fields=len(fields),
            fields=fields,
            raw_columns=raw_columns,
            warnings=warnings,
        )
