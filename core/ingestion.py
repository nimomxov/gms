"""
GMS Core — Data Ingestion Module
Validates, normalizes, and prepares raw CSV scan data.
"""

import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("gms.ingestion")


@dataclass
class ScanDataset:
    """Validated, normalized scan dataset."""
    scan_id: str
    raw_df: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class DataIngestionError(ValueError):
    pass


class ScanIngestionEngine:
    """
    Loads and validates geophysical scan CSV files.
    Expected CSV columns: x, y, value  (at minimum)
    Value range: configurable (default 0–1024 for ADC output)
    """

    def __init__(self, config: dict):
        self.cfg = config.get("data_ingestion", {})
        self.expected_cols = self.cfg.get("expected_columns", ["x", "y", "value"])
        self.value_range = self.cfg.get("value_range", [0, 1024])
        self.min_samples = self.cfg.get("min_samples_per_scan", 50)
        self.max_gap = self.cfg.get("max_gap_fraction", 0.2)

    def load_csv(self, filepath: str | Path) -> ScanDataset:
        """Load a single CSV scan file with full validation."""
        path = Path(filepath)
        if not path.exists():
            raise DataIngestionError(f"File not found: {path}")

        logger.info(f"Loading scan: {path.name}")
        warnings = []

        try:
            df = pd.read_csv(path, comment="#")
        except Exception as e:
            raise DataIngestionError(f"CSV parse error: {e}")

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Check required columns
        missing = [c for c in self.expected_cols if c not in df.columns]
        if missing:
            raise DataIngestionError(f"Missing columns: {missing}")

        df = df[self.expected_cols].copy()
        n_raw = len(df)

        # Drop NaN rows
        df.dropna(inplace=True)
        n_dropped = n_raw - len(df)
        gap_frac = n_dropped / max(n_raw, 1)

        if gap_frac > self.max_gap:
            raise DataIngestionError(
                f"Too many missing values: {gap_frac:.1%} > {self.max_gap:.1%}"
            )
        if n_dropped > 0:
            warnings.append(f"Dropped {n_dropped} rows with NaN values ({gap_frac:.1%})")

        if len(df) < self.min_samples:
            raise DataIngestionError(
                f"Too few samples: {len(df)} < {self.min_samples}"
            )

        # Value range check
        vmin, vmax = self.value_range
        out_of_range = df[(df["value"] < vmin) | (df["value"] > vmax)]
        if len(out_of_range) > 0:
            frac = len(out_of_range) / len(df)
            if frac > 0.05:
                warnings.append(
                    f"{frac:.1%} of values outside expected range [{vmin}, {vmax}]"
                )
            # Clip to range
            df["value"] = df["value"].clip(vmin, vmax)

        # Compute scan_id from file hash
        content_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        scan_id = f"{path.stem}_{content_hash}"

        metadata = {
            "source_file": path.name,
            "n_samples": len(df),
            "x_range": [float(df["x"].min()), float(df["x"].max())],
            "y_range": [float(df["y"].min()), float(df["y"].max())],
            "value_mean": float(df["value"].mean()),
            "value_std": float(df["value"].std()),
        }

        logger.info(
            f"Scan {scan_id}: {len(df)} samples, "
            f"x∈[{metadata['x_range'][0]:.2f},{metadata['x_range'][1]:.2f}], "
            f"y∈[{metadata['y_range'][0]:.2f},{metadata['y_range'][1]:.2f}]"
        )

        if warnings:
            for w in warnings:
                logger.warning(f"  [{scan_id}] {w}")

        return ScanDataset(
            scan_id=scan_id,
            raw_df=df,
            x=df["x"].values.astype(np.float64),
            y=df["y"].values.astype(np.float64),
            values=df["value"].values.astype(np.float64),
            metadata=metadata,
            warnings=warnings,
        )

    def load_multiple(self, filepaths: list) -> list[ScanDataset]:
        """Load multiple CSV scans. Skips failed files with logging."""
        datasets = []
        for fp in filepaths:
            try:
                ds = self.load_csv(fp)
                datasets.append(ds)
            except DataIngestionError as e:
                logger.error(f"Skipping {fp}: {e}")
        logger.info(f"Loaded {len(datasets)}/{len(filepaths)} scans successfully")
        return datasets
