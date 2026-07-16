"""
GMS — Pipeline Orchestrator  v2.0

Assembles the 3 decoupled stages into a reproducible pipeline:
  1. InterpolatorBase  → GriddedScan
  2. BaselineRemoverBase → BaselinedGrid
  3. AnomalyDetectorBase → DetectionResult

Key features:
  - Compatibility checking: warns before running known bad combinations
  - PipelineConfig: fully describes a pipeline for reproducibility
  - Experiment comparison: run multiple configs, compare results
  - No tight coupling: stages are swapped via config, not code

Preset pipelines:
  "stable"      → cubic + line_median + log        (default, production-ready)
  "rbf"         → rbf_thin_plate + none + amplitude (experimental, physics-aware)
  "sensitive"   → cubic + line_median + hybrid      (higher recall, more FP)
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation

from .abstractions import (
    RawScan, GriddedScan, BaselinedGrid, DetectionResult, RawAnomaly
)
from .interpolators.plugins import get_interpolator, INTERPOLATOR_REGISTRY
from .baselines.plugins import get_baseline, BASELINE_REGISTRY
from .detectors.plugins import get_detector, DETECTOR_REGISTRY
from .ingestion import ScanIngestionEngine, DataIngestionError
from .reliability import ScanReliabilityEngine, ScanReliability

logger = logging.getLogger("gms.pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# PipelineConfig — fully describes one experiment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Complete, reproducible description of a GMS analysis pipeline.
    Serializes to/from YAML and JSON.
    Includes a hash for exact experiment matching.
    """
    # Stage selection
    interpolator: str = "cubic"
    baseline: str = "line_median"
    detector: str = "log"

    # Stage parameters
    interpolator_params: dict = field(default_factory=dict)
    baseline_params: dict = field(default_factory=dict)
    detector_params: dict = field(default_factory=dict)

    # Grid
    resolution: float = 0.1

    # Pre-processing
    global_dc_remove: bool = True
    per_line_dc_remove: bool = True
    mad_threshold: float = 3.5
    gauss_kernel: float = 3.0     # post-grid light smoothing kernel size

    # Detection thresholds
    snr_min: float = 2.6
    min_spatial_extent: int = 5
    multi_scale_sigmas: list = field(default_factory=lambda: [1.0, 2.0, 4.0])

    # Classification rules (passed through to detector)
    classification: dict = field(default_factory=dict)

    # Decision rules
    decision: dict = field(default_factory=lambda: {
        "DIG":    {"min_scan_confirmations": 2, "min_confidence": 0.70,
                   "max_uncertainty": 0.25, "snr_min": 4.0},
        "RESCAN": {"min_scan_confirmations": 1, "min_confidence": 0.45},
    })

    # Metadata
    name: str = "unnamed"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def config_hash(self) -> str:
        """SHA-256 of all parameters — unique fingerprint for this exact config."""
        d = asdict(self)
        d.pop("created_at", None)  # exclude timestamp from hash
        canonical = json.dumps(d, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=True)

    @classmethod
    def from_yaml(cls, text: str) -> "PipelineConfig":
        data = yaml.safe_load(text)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_yaml(Path(path).read_text())

    def to_analysis_config(self) -> dict:
        """Convert to the dict format expected by detector + classifier."""
        return {
            "anomaly_detection": {
                "snr_min": self.snr_min,
                "min_spatial_extent": self.min_spatial_extent,
                "multi_scale_sigmas": self.multi_scale_sigmas,
            },
            "classification": self.classification,
            "decision": self.decision,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Preset pipelines
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, PipelineConfig] = {
    "stable": PipelineConfig(
        name="stable",
        description="Production default. cubic+line_median+LoG. Tested and stable.",
        interpolator="cubic",
        baseline="line_median",
        detector="log",
        baseline_params={"window_fraction": 0.90},
        snr_min=2.6, min_spatial_extent=5,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.48, "snr_min": 3.5, "coherence_min": 0.55},
            "CAVITY":        {"final_score_max": 0.42, "snr_min": 3.0, "smoothness_min": 0.65},
            "ROCK_DEBRIS":   {"snr_min": 2.5},
            "NOISE":         {"snr_max": 1.8},
        },
    ),
    "stable_v2": PipelineConfig(
        name="stable_v2",
        description=(
            "stable + TopologyValidator cascade + GlobalReliability. "
            "Fixes rock_debris FP and noise_only FP. Lower FPR than stable."
        ),
        interpolator="cubic",
        baseline="line_median",
        detector="cascaded_matched",
        baseline_params={"window_fraction": 0.90},
        detector_params={
            "ncc_threshold_wide": 0.30,
            "n_depths": 4, "n_orientations": 4,
            "template_size": 13,
            "use_amplitude_primary": False,  # LoG primary — works with line_median
        },
        snr_min=2.4, min_spatial_extent=5,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.44, "snr_min": 3.0, "coherence_min": 0.48},
            "CAVITY":        {"final_score_max": 0.44, "snr_min": 2.5, "smoothness_min": 0.60},
            "ROCK_DEBRIS":   {"snr_min": 2.2},
            "NOISE":         {"snr_max": 1.8},
        },
        decision={
            "DIG":    {"min_scan_confirmations": 2, "min_confidence": 0.55,
                       "max_uncertainty": 0.30, "snr_min": 3.5},
            "RESCAN": {"min_scan_confirmations": 1, "min_confidence": 0.38},
        },
    ),
    "rbf": PipelineConfig(
        name="rbf",
        description="Experimental. RBF+NoBaseline+Amplitude. Physics-aware interpolation.",
        interpolator="rbf_thin_plate",
        baseline="none",
        detector="amplitude",
        detector_params={"mask_erosion": 4},
        snr_min=2.6, min_spatial_extent=6,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.48, "snr_min": 3.5, "coherence_min": 0.55},
            "CAVITY":        {"final_score_max": 0.42, "snr_min": 3.0, "smoothness_min": 0.65},
            "ROCK_DEBRIS":   {"snr_min": 2.5},
            "NOISE":         {"snr_max": 1.8},
        },
    ),
    "sensitive": PipelineConfig(
        name="sensitive",
        description="Higher recall. cubic+line_median+Hybrid. More false positives.",
        interpolator="cubic",
        baseline="line_median",
        detector="hybrid",
        detector_params={"mask_erosion": 2},
        snr_min=2.2, min_spatial_extent=4,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.45, "snr_min": 3.0, "coherence_min": 0.50},
            "CAVITY":        {"final_score_max": 0.45, "snr_min": 2.5, "smoothness_min": 0.60},
            "ROCK_DEBRIS":   {"snr_min": 2.0},
            "NOISE":         {"snr_max": 1.5},
        },
    ),
    "matched": PipelineConfig(
        name="matched",
        description="Matched dipole filter + adaptive threshold. Best physics-based detection.",
        interpolator="cubic",
        baseline="adaptive_local",
        detector="matched_dipole",
        baseline_params={"window_fraction": 0.90, "local_window_cells": 12},
        detector_params={"ncc_threshold": 0.40, "n_depths": 4, "n_orientations": 4},
        snr_min=2.4, min_spatial_extent=5,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.45, "snr_min": 3.0, "coherence_min": 0.50},
            "CAVITY":        {"final_score_max": 0.45, "snr_min": 2.5, "smoothness_min": 0.60},
            "ROCK_DEBRIS":   {"snr_min": 2.0},
            "NOISE":         {"snr_max": 1.8},
        },
    ),
    "matched_v2": PipelineConfig(
        name="matched_v2",
        description=(
            "Cascaded validation + multiscale baseline. "
            "LoG-primary detection, NCC as validation boost. "
            "Best cavity + metal coverage, moderate FPR."
        ),
        interpolator="cubic",
        baseline="multiscale",
        detector="cascaded_matched",
        baseline_params={"scales_cells": [3, 8, 18], "pre_line_dc": True},
        detector_params={
            "ncc_threshold_wide": 0.20,
            "n_depths": 4, "n_orientations": 4,
            "template_size": 13,
            "use_amplitude_primary": True,  # required for multiscale baseline
        },
        snr_min=2.2, min_spatial_extent=5,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.42, "snr_min": 2.5, "coherence_min": 0.45},
            "CAVITY":        {"final_score_max": 0.46, "snr_min": 2.2, "smoothness_min": 0.55},
            "ROCK_DEBRIS":   {"snr_min": 1.8},
            "NOISE":         {"snr_max": 1.6},
        },
        decision={
            "DIG":    {"min_scan_confirmations": 2, "min_confidence": 0.50,
                       "max_uncertainty": 0.40, "snr_min": 2.8},
            "RESCAN": {"min_scan_confirmations": 1, "min_confidence": 0.30},
        },
    ),
    "matched": PipelineConfig(
        name="matched",
        description="Matched dipole filter + adaptive local threshold. Best physics-based detection.",
        interpolator="cubic",
        baseline="adaptive_local",
        detector="matched_dipole",
        baseline_params={"window_fraction": 0.90, "local_window_cells": 12},
        detector_params={"ncc_threshold": 0.40, "n_depths": 4, "n_orientations": 4},
        snr_min=2.4, min_spatial_extent=5,
        classification={
            "FERROUS_METAL": {"final_score_min": 0.45, "snr_min": 3.0, "coherence_min": 0.50},
            "CAVITY":        {"final_score_max": 0.45, "snr_min": 2.5, "smoothness_min": 0.60},
            "ROCK_DEBRIS":   {"snr_min": 2.0},
            "NOISE":         {"snr_max": 1.8},
        },
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility checker
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_BAD_COMBOS = [
    (
        {"rbf_thin_plate", "rbf"},
        {"log_detector", "log"},
        "RBF produces C∞ surfaces → LoG curvature collapses → anomalies missed. "
        "Use detector: amplitude or hybrid with RBF."
    ),
    (
        {"rbf_thin_plate", "rbf"},
        {"line_median", "highpass"},
        "Post-grid baseline after RBF causes double-subtraction → anomaly absorption. "
        "Use baseline: none with RBF."
    ),
]

def check_compatibility(cfg: PipelineConfig) -> list[str]:
    """Returns list of warning strings for known bad combos."""
    warnings = []
    interp  = {cfg.interpolator}
    base    = {cfg.baseline}
    detect  = {cfg.detector}

    for bad_interp, bad_other, msg in KNOWN_BAD_COMBOS:
        if interp & bad_interp:
            if base & bad_other or detect & bad_other:
                warnings.append(f"⚠️ INCOMPATIBLE COMBO: {msg}")

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class GMSPipeline:
    """
    Assembles and runs the 3-stage GMS analysis pipeline.
    Stages are fully decoupled — swap any stage via PipelineConfig.
    """

    def __init__(self, pipeline_cfg: PipelineConfig, gms_config: dict):
        self.cfg     = pipeline_cfg
        self.gms_cfg = gms_config  # full system config (scan limits etc.)

        # Check compatibility before building
        self.compat_warnings = check_compatibility(pipeline_cfg)
        for w in self.compat_warnings:
            logger.warning(w)

        # Instantiate stages
        self.interpolator = get_interpolator(pipeline_cfg.interpolator)
        self.baseline     = get_baseline(pipeline_cfg.baseline,
                                          pipeline_cfg.baseline_params)
        self.detector     = get_detector(pipeline_cfg.detector,
                                          pipeline_cfg.detector_params)

        self.analysis_cfg = pipeline_cfg.to_analysis_config()

        self.reliability_engine = ScanReliabilityEngine()
        logger.info(
            f"Pipeline [{pipeline_cfg.name}] assembled: "
            f"{self.interpolator.name} → {self.baseline.name} → {self.detector.name}"
        )

    # ── Per-scan processing ───────────────────────────────────────────────────

    def process_scan(self, filepath: str) -> tuple[BaselinedGrid, DetectionResult]:
        """Run full pipeline on one CSV file."""
        from scipy.stats import median_abs_deviation as mad

        # 1. Ingest
        ingestion = ScanIngestionEngine(self.gms_cfg)
        dataset   = ingestion.load_csv(filepath)

        x, y, v = dataset.x.copy(), dataset.y.copy(), dataset.values.copy()

        # 2. Pre-processing (always applied, independent of pipeline stages)
        if self.cfg.global_dc_remove:
            v = v - float(np.median(v))

        if self.cfg.per_line_dc_remove:
            for y_val in np.unique(y):
                mask = y == y_val
                v[mask] = v[mask] - np.median(v[mask])

        # MAD outlier rejection
        zs = _mad_zscore(v)
        valid = np.abs(zs) < self.cfg.mad_threshold
        x, y, v = x[valid], y[valid], v[valid]

        raw_scan = RawScan(
            scan_id=dataset.scan_id,
            x=x, y=y, values=v,
            metadata=dataset.metadata,
            warnings=dataset.warnings,
        )

        # 3. Interpolate
        gridded = self.interpolator.interpolate(raw_scan, self.cfg.resolution)

        # 4. Light Gaussian smoothing (always, after interpolation)
        sigma = self.cfg.gauss_kernel / 4.0
        gz_smooth = ndimage.gaussian_filter(gridded.grid_z, sigma=sigma)
        gz_smooth[~gridded.grid_mask] = 0.0
        gridded = GriddedScan(
            **{**gridded.__dict__, "grid_z": gz_smooth}
        )

        # 5. Baseline removal
        baselined = self.baseline.remove(gridded)

        # 6. Detect
        result = self.detector.detect(baselined, self.analysis_cfg)

        # 7. Reliability assessment + penalty
        reliability = self.reliability_engine.assess(baselined)
        result.anomalies = self.reliability_engine.apply_penalty(
            result.anomalies, reliability
        )
        # Store reliability in warnings if poor
        if not reliability.is_reliable:
            result.warnings.append(
                f"LOW RELIABILITY ({reliability.quality_label}): {reliability.message}"
            )
        result.__dict__['reliability'] = reliability

        return baselined, result

    # ── Full session ──────────────────────────────────────────────────────────

    def run_session(self, filepaths: list[str],
                    session_id: str = "session") -> dict:
        """
        Run pipeline on all scans, cross-validate, make decision.
        Returns serializable result dict.
        """
        from .decision_engine import CrossScanValidator
        from viz.visualization import GeoVizEngine

        grids, results = [], []
        skipped = []

        for fp in filepaths:
            try:
                grid, det = self.process_scan(fp)
                grids.append(grid)
                results.append(det)
                logger.info(
                    f"  [{det.scan_id[:18]}] {len(det.anomalies)} anomalies  "
                    f"quality={det.scan_quality_score:.2f}  "
                    f"detector={det.detector_name}"
                )
            except Exception as e:
                logger.error(f"  Skipping {fp}: {e}")
                skipped.append(str(fp))

        if not results:
            raise RuntimeError("All scan files failed processing")

        # Cross-scan validation
        validator = CrossScanValidator({"decision": self.cfg.decision})
        # Adapt DetectionResult to old format for reuse
        report = validator.validate(
            _adapt_detection_results(results),
            session_id=session_id
        )

        # Visualize
        viz = GeoVizEngine(self.gms_cfg, output_dir="reports")
        heatmap_paths = {}
        for grid, det in zip(grids, results):
            det_old = _adapt_to_old_detection(det)
            grid_old = _adapt_to_old_grid(grid)
            paths = viz.render_scan_heatmap(
                grid_old, det_old,
                output_prefix=f"{session_id}_{grid.scan_id}"
            )
            heatmap_paths[grid.scan_id] = paths

        # Final overview map
        viz.render_final_report_map(
            [_adapt_to_old_grid(g) for g in grids],
            report,
            output_prefix=f"{session_id}_final"
        )

        return _serialize_session(report, session_id, self.cfg, skipped, heatmap_paths)

    # ── Experiment comparison ─────────────────────────────────────────────────

    @classmethod
    def compare_presets(cls, filepaths: list[str],
                         presets: list[str],
                         gms_config: dict,
                         session_id: str = "compare") -> dict:
        """
        Run multiple pipeline presets on the same scans.
        Returns comparison dict keyed by preset name.
        """
        comparison = {}
        for preset_name in presets:
            if preset_name not in PRESETS:
                logger.warning(f"Unknown preset '{preset_name}' — skipping")
                continue
            cfg = PRESETS[preset_name]
            logger.info(f"\n{'='*50}\nRunning preset: {preset_name}\n{'='*50}")
            try:
                pipeline = cls(cfg, gms_config)
                result   = pipeline.run_session(
                    filepaths,
                    session_id=f"{session_id}_{preset_name}"
                )
                comparison[preset_name] = {
                    "decision": result["decision"],
                    "n_confirmed": result["confidence_summary"]["n_confirmed"],
                    "overall_confidence": result["confidence_summary"]["overall"],
                    "config_hash": cfg.config_hash(),
                    "pipeline": f"{cfg.interpolator}+{cfg.baseline}+{cfg.detector}",
                }
            except Exception as e:
                comparison[preset_name] = {"error": str(e)}

        return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mad_zscore(values):
    from scipy.stats import median_abs_deviation as mad
    med = np.median(values)
    m   = mad(values, nan_policy="omit")
    if m < 1e-10: return np.zeros_like(values)
    return 0.6745 * (values - med) / m


def _adapt_detection_results(results: list[DetectionResult]):
    """Wrap new DetectionResult to be compatible with old CrossScanValidator."""
    from .anomaly_detection import Anomaly as OldAnomaly
    from .anomaly_detection import DetectionResult as OldResult

    old_results = []
    for r in results:
        old_anomalies = []
        for a in r.anomalies:
            old_a = OldAnomaly(
                anomaly_id=a.anomaly_id,
                cx=a.cx, cy=a.cy,
                marker_cx=a.marker_cx, marker_cy=a.marker_cy,
                extent_cells=a.extent_cells,
                peak_amplitude=a.peak_amplitude,
                snr_robust=a.snr_robust,
                smoothness_score=a.smoothness_score,
                dipole_score=a.dipole_score,
                polarity_ratio=a.polarity_ratio,
                spatial_coherence=a.spatial_coherence,
                final_score=a.final_score,
                uncertainty=a.uncertainty,
                raw_label=a.raw_label,
                confidence=a.confidence,
                bbox=a.bbox,
            )
            old_anomalies.append(old_a)
        old_r = OldResult(
            scan_id=r.scan_id,
            anomalies=old_anomalies,
            scan_quality_score=r.scan_quality_score,
            noise_floor=r.noise_floor,
            warnings=r.warnings,
        )
        old_results.append(old_r)
    return old_results


def _adapt_to_old_detection(det: DetectionResult):
    """Adapt new DetectionResult for visualization (uses old Anomaly type)."""
    from .anomaly_detection import Anomaly as OldAnomaly
    from .anomaly_detection import DetectionResult as OldResult

    old_anomalies = []
    for a in det.anomalies:
        old_a = OldAnomaly(
            anomaly_id=a.anomaly_id,
            cx=a.cx, cy=a.cy,
            marker_cx=a.marker_cx, marker_cy=a.marker_cy,
            extent_cells=a.extent_cells,
            peak_amplitude=a.peak_amplitude,
            snr_robust=a.snr_robust,
            smoothness_score=a.smoothness_score,
            dipole_score=a.dipole_score,
            polarity_ratio=a.polarity_ratio,
            spatial_coherence=a.spatial_coherence,
            final_score=a.final_score,
            uncertainty=a.uncertainty,
            raw_label=a.raw_label,
            confidence=a.confidence,
            bbox=a.bbox,
        )
        old_anomalies.append(old_a)
    return OldResult(
        scan_id=det.scan_id,
        anomalies=old_anomalies,
        scan_quality_score=det.scan_quality_score,
        noise_floor=det.noise_floor,
        warnings=det.warnings,
    )


def _adapt_to_old_grid(grid: BaselinedGrid):
    """Adapt new BaselinedGrid for visualization (uses old ProcessedGrid type)."""
    from .signal_processing import ProcessedGrid as OldGrid
    return OldGrid(
        scan_id=grid.scan_id,
        grid_z=grid.grid_z,
        grid_x=grid.grid_x,
        grid_y=grid.grid_y,
        grid_mask=grid.grid_mask,
        baseline=0.0,
        noise_floor=grid.noise_floor,
        dynamic_range=grid.dynamic_range,
        drift_method=grid.baseline_name,
        interp_method=grid.interp_name,
        warnings=grid.warnings,
    )


def _serialize_session(report, session_id, cfg, skipped, heatmap_paths) -> dict:
    confirmed = [{
        "group_id": c.group_id,
        "label": c.best_label,
        "centroid_x": c.centroid_x,
        "centroid_y": c.centroid_y,
        "combined_confidence": c.combined_confidence,
        "scan_confirmations": c.scan_confirmations,
        "mean_snr": c.mean_snr,
        "mean_uncertainty": c.mean_uncertainty,
        "label_agreement": c.label_agreement,
    } for c in report.confirmed_anomalies]

    return {
        "session_id": session_id,
        "pipeline": {
            "name": cfg.name,
            "interpolator": cfg.interpolator,
            "baseline": cfg.baseline,
            "detector": cfg.detector,
            "config_hash": cfg.config_hash(),
        },
        "decision": report.decision,
        "confidence_summary": report.confidence_summary,
        "anomalies": confirmed,
        "scan_quality": report.scan_quality,
        "warnings": report.warnings + (["Skipped: " + s for s in skipped] if skipped else []),
        "n_scans_processed": report.n_scans_processed,
        "heatmap_paths": heatmap_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(gms_config: dict,
                    preset: str = None,
                    pipeline_section: dict = None) -> GMSPipeline:
    """
    Build a GMSPipeline from either a preset name or a config section.

    Priority:
      1. preset (if given) → use PRESETS[preset]
      2. pipeline_section (from YAML config) → build PipelineConfig
      3. fallback → "stable" preset

    Example pipeline_section in gms_config.yaml:
      pipeline:
        name: my_custom
        interpolator: cubic
        baseline: line_median
        detector: log
        baseline_params:
          window_fraction: 0.90
        snr_min: 2.6
    """
    if preset and preset in PRESETS:
        cfg = PRESETS[preset]
        logger.info(f"Using preset pipeline: {preset}")
        return GMSPipeline(cfg, gms_config)

    if pipeline_section:
        fields = PipelineConfig.__dataclass_fields__
        kwargs = {k: v for k, v in pipeline_section.items() if k in fields}
        cfg = PipelineConfig(**kwargs)
        logger.info(f"Using config-defined pipeline: {cfg.name}")
        return GMSPipeline(cfg, gms_config)

    logger.info("Using default preset: stable")
    return GMSPipeline(PRESETS["stable"], gms_config)
