"""
GMS — AdaptiveIngestionEngine  v2.3

Replaces the rigid ScanIngestionEngine with a capability-aware ingestion
pipeline that:

  1. Inspects raw CSV fields (CSVInspector)
  2. Detects device profile (DeviceProfileRegistry) — or falls back to auto
  3. Maps semantic roles (SemanticFieldMapper)
  4. Extracts device capabilities (CapabilityExtractor)
  5. Converts rows to TelemetryFrames
  6. Builds a RawScan for the main pipeline

The resulting AdaptiveScanDataset carries:
  - TelemetryFrames (unified internal model)
  - DeviceCapabilities (drives all downstream gating)
  - ComposedPipeline (which stages are active)
  - SemanticMapping (for debug / logging)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema.inspector import CSVInspector
from .schema.mapper import SemanticFieldMapper
from .schema.capabilities import (
    CapabilityExtractor, DeviceCapabilities, TelemetryFrame, TelemetryGrade
)
from .device_profiles import DeviceProfileRegistry
from .pipeline_composer import DynamicPipelineComposer, ComposedPipeline
from .abstractions import RawScan

logger = logging.getLogger("gms.adaptive_ingestion")


@dataclass
class AdaptiveScanDataset:
    """
    Full scan dataset with resolved capabilities and composed pipeline.
    This is the primary input to all downstream processing.
    """
    scan_id: str
    source: str
    frames: list[TelemetryFrame]
    capabilities: DeviceCapabilities
    pipeline: ComposedPipeline
    raw_scan: RawScan           # compatible with existing pipeline stages
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    device_profile_name: Optional[str] = None

    @property
    def grade(self) -> TelemetryGrade:
        return self.capabilities.grade

    @property
    def n_samples(self) -> int:
        return len(self.frames)

    def signals(self) -> np.ndarray:
        return np.array([f.signal for f in self.frames])

    def positions(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.capabilities.has_position:
            return None, None
        x = np.array([f.x for f in self.frames if f.x is not None])
        y = np.array([f.y for f in self.frames if f.y is not None])
        return x, y

    def snr_array(self) -> Optional[np.ndarray]:
        if not self.capabilities.has_snr:
            return None
        return np.array([f.snr if f.snr is not None else np.nan for f in self.frames])


class AdaptiveIngestionEngine:
    """
    Capability-aware CSV ingestion.

    Usage:
        engine = AdaptiveIngestionEngine()
        dataset = engine.load("scan_A.csv")
        # dataset.capabilities tells you what the device supports
        # dataset.pipeline tells you which stages are active
    """

    def __init__(
        self,
        profiles_dir: Optional[str | Path] = None,
        extra_aliases: Optional[dict[str, str]] = None,
        min_samples: int = 10,
    ):
        self.inspector = CSVInspector()
        self.profile_registry = DeviceProfileRegistry(profiles_dir)
        self.capability_extractor = CapabilityExtractor()
        self.pipeline_composer = DynamicPipelineComposer()
        self.min_samples = min_samples
        self._extra_aliases = extra_aliases or {}

    def load(self, filepath: str | Path) -> AdaptiveScanDataset:
        """Full capability-aware load of a CSV scan file."""
        path = Path(filepath)
        warnings: list[str] = []

        # ── Step 1: Load raw CSV ─────────────────────────────────────────────
        try:
            df = pd.read_csv(path, comment="#")
        except Exception as e:
            raise ValueError(f"AdaptiveIngestion: failed to read '{path.name}': {e}")

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        n_raw = len(df)

        # ── Step 2: Inspect fields ───────────────────────────────────────────
        inventory = self.inspector.inspect_dataframe(df, source=path.name)
        warnings.extend(inventory.warnings)

        # ── Step 3: Detect device profile ────────────────────────────────────
        profile = self.profile_registry.detect(inventory.field_names())
        profile_name = profile.name if profile else "auto_detection"

        extra_aliases = dict(self._extra_aliases)
        if profile:
            extra_aliases.update(profile.field_aliases)

        # ── Step 4: Semantic mapping ─────────────────────────────────────────
        mapper = SemanticFieldMapper(extra_aliases=extra_aliases)
        mapping = mapper.map(inventory)
        warnings.extend(mapping.warnings)

        if not mapping.has_role("signal"):
            raise ValueError(
                f"AdaptiveIngestion: no signal field found in '{path.name}'. "
                f"Available fields: {inventory.field_names()}"
            )

        # ── Step 5: Extract capabilities ─────────────────────────────────────
        cap = self.capability_extractor.extract(mapping)

        # ── Step 6: Compose pipeline ─────────────────────────────────────────
        composed = self.pipeline_composer.compose(cap)
        warnings.extend(composed.warnings)

        # ── Step 7: Drop NaN rows, validate sample count ─────────────────────
        signal_col = mapping.get_column("signal")
        df = df.dropna(subset=[signal_col])

        if len(df) < self.min_samples:
            raise ValueError(
                f"AdaptiveIngestion: too few valid samples ({len(df)} < {self.min_samples}) "
                f"in '{path.name}'"
            )
        if n_raw - len(df) > 0:
            pct = (n_raw - len(df)) / n_raw
            warnings.append(f"Dropped {n_raw - len(df)} rows with NaN signal ({pct:.0%})")

        # ── Step 8: Build TelemetryFrames ────────────────────────────────────
        frames = TelemetryFrame.from_dataframe(df, mapping)

        if len(frames) < self.min_samples:
            raise ValueError(
                f"AdaptiveIngestion: only {len(frames)} valid TelemetryFrames produced "
                f"from '{path.name}'"
            )

        # ── Step 9: Build RawScan (backward-compatible) ──────────────────────
        signals = np.array([f.signal for f in frames], dtype=np.float64)

        if cap.has_position:
            xs = np.array([f.x if f.x is not None else np.nan for f in frames])
            ys = np.array([f.y if f.y is not None else np.nan for f in frames])
        else:
            # Synthesize a 1D line arrangement for non-positional devices
            xs = np.arange(len(frames), dtype=np.float64)
            ys = np.zeros(len(frames), dtype=np.float64)
            warnings.append(
                "No x/y position available — synthesized 1D line arrangement. "
                "2D heatmap and interpolation are disabled."
            )

        import hashlib
        scan_id = f"{path.stem}_{hashlib.md5(path.read_bytes()).hexdigest()[:8]}"

        raw_scan = RawScan(
            scan_id=scan_id,
            x=xs,
            y=ys,
            values=signals,
            metadata={
                "source": path.name,
                "grade": cap.grade.name,
                "profile": profile_name,
                "n_samples": len(frames),
                "has_position": cap.has_position,
                "has_snr": cap.has_snr,
                "has_heading": cap.has_heading,
            },
            warnings=warnings,
        )

        metadata = {
            "source_file": path.name,
            "n_samples": len(frames),
            "profile": profile_name,
            "grade": cap.grade.name,
            "grade_label": cap.grade.label(),
            "active_stages": composed.enabled_stages,
            "disabled_stages": list(composed.disabled_stages.keys()),
        }

        logger.info(
            f"[AdaptiveIngestion] '{path.name}' → "
            f"grade={cap.grade.name} profile={profile_name} "
            f"samples={len(frames)} stages={len(composed.enabled_stages)} active"
        )

        if composed.disabled_stages:
            for stage, reason in composed.disabled_stages.items():
                logger.info(f"  [DISABLED] {stage}: {reason}")

        return AdaptiveScanDataset(
            scan_id=scan_id,
            source=path.name,
            frames=frames,
            capabilities=cap,
            pipeline=composed,
            raw_scan=raw_scan,
            metadata=metadata,
            warnings=warnings,
            device_profile_name=profile_name,
        )

    def load_multiple(self, filepaths: list) -> list[AdaptiveScanDataset]:
        """Load multiple files, skipping any that fail."""
        datasets = []
        for fp in filepaths:
            try:
                ds = self.load(fp)
                datasets.append(ds)
            except ValueError as e:
                logger.error(f"Skipping '{fp}': {e}")
        logger.info(
            f"[AdaptiveIngestion] Loaded {len(datasets)}/{len(filepaths)} scans"
        )
        return datasets
