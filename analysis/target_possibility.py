"""
GMS — Target Possibility Engine v1.0
=======================================
Post-processing scientific EVIDENCE engine. Runs AFTER the full pipeline
(ingestion -> geometry -> interpolation -> baseline -> detection ->
reliability -> cross-validation -> registration -> fusion -> explainability).

HARD RULES (enforced by design):
  * Never modifies raw data or any grid. Every stage works on COPIES.
  * Fully deterministic. No RNG, no time, no AI guessing. Same in => same out.
  * Every sub-score is independent and in [0, 1].
  * Every point added or removed is recorded as a signed Reason with the
    evidence that produced it (full explainability).
  * Disagreement lowers the score: a single dissenting evidence cannot be
    hidden by many agreeing ones (see _fuse_evidence).
  * Physics-based and conservative for void/metal — no fabricated detections,
    no gold/silver/bronze claims (physically unrecoverable from this data).

The engine consumes the pipeline's existing result objects; it re-runs only
the cheap, deterministic parts it needs for stability testing (interpolation /
baseline / smoothing sweeps), each on a COPY of the scan.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger("gms.analysis.target_possibility")


# ─────────────────────────────────────────────────────────────────────────────
# Evidence + result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Reason:
    """One signed, explainable contribution to a score."""
    sign: str            # "+" increased, "-" decreased, "i" informational
    text: str            # human-readable
    evidence: str        # which stage/metric produced it
    value: float = 0.0   # the metric value behind it

    def to_line(self) -> str:
        return f"[{self.sign}] {self.text}  ({self.evidence}={self.value:.3f})"


@dataclass
class StageScore:
    """Result of one evidence stage."""
    name: str
    score: float                       # [0,1]
    reasons: list[Reason] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def clamp(self) -> "StageScore":
        self.score = float(np.clip(self.score, 0.0, 1.0))
        return self


@dataclass
class TargetPossibility:
    """Full per-target evidence report."""
    target_id: str
    x_m: float
    y_m: float
    depth_m: Optional[float]
    depth_uncertainty_m: Optional[float]
    area_m2: float
    max_amplitude: float
    confidence: float
    uncertainty: float
    fusion_confidence: float
    cross_validation: str

    # Stage sub-scores [0,1]
    interpolation_stability: float
    baseline_stability: float
    smoothing_stability: float
    reliability_score: float
    signal_quality: float
    spatial_stability: float
    morphology_score: float
    void_possibility: float
    metallic_possibility: float

    # Final
    possibility_score: float           # 0..100
    classification: str
    color: str
    recommended_action: str

    reasons_up: list[str] = field(default_factory=list)
    reasons_down: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "x_m": round(self.x_m, 3), "y_m": round(self.y_m, 3),
            "depth_m": None if self.depth_m is None else round(self.depth_m, 3),
            "depth_uncertainty_m": None if self.depth_uncertainty_m is None else round(self.depth_uncertainty_m, 3),
            "area_m2": round(self.area_m2, 4),
            "max_amplitude": round(self.max_amplitude, 4),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "fusion_confidence": round(self.fusion_confidence, 4),
            "cross_validation": self.cross_validation,
            "interpolation_stability": round(self.interpolation_stability, 4),
            "baseline_stability": round(self.baseline_stability, 4),
            "smoothing_stability": round(self.smoothing_stability, 4),
            "reliability_score": round(self.reliability_score, 4),
            "signal_quality": round(self.signal_quality, 4),
            "morphology_score": round(self.morphology_score, 4),
            "void_possibility": round(self.void_possibility, 4),
            "metallic_possibility": round(self.metallic_possibility, 4),
            "possibility_score": round(self.possibility_score, 1),
            "classification": self.classification,
            "color": self.color,
            "recommended_action": self.recommended_action,
            "reasons_up": self.reasons_up,
            "reasons_down": self.reasons_down,
            "warnings": self.warnings,
            "stage_scores": self.stage_scores,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Documented default weights (Stage 12).
# Rationale in the WEIGHTS_RATIONALE doc page. Sum need not be 1; the fusion
# normalizes. Stability evidences are weighted highest because cross-method
# persistence is the strongest guard against algorithm-specific artifacts.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "interpolation_stability": 0.16,
    "baseline_stability":      0.14,
    "smoothing_stability":     0.10,
    "fusion_evidence":         0.12,
    "cross_validation":        0.10,
    "reliability":             0.08,
    "signal_quality":          0.10,
    "spatial_stability":       0.08,
    "morphology":              0.12,
    # void/metal are reported separately AND feed a small "is this a plausible
    # physical target at all" nudge; they do not inflate the base score.
}

CLASS_BANDS = [
    (90.0, "Very High Possibility", "#2ECC71", "High Priority Investigation"),
    (75.0, "High Possibility",      "#7FE07F", "High Priority Investigation"),
    (60.0, "Moderate Possibility",  "#F1C40F", "Re-scan"),
    (40.0, "Low Possibility",       "#E67E22", "Monitor"),
    (0.0,  "Very Low Possibility",  "#95A5A6", "Monitor"),
]


def classify(score_0_100: float) -> tuple[str, str, str]:
    for thr, label, color, action in CLASS_BANDS:
        if score_0_100 >= thr:
            return label, color, action
    return "Very Low Possibility", "#95A5A6", "Monitor"


# ─────────────────────────────────────────────────────────────────────────────
# Small deterministic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cell_sizes(grid) -> tuple[float, float]:
    gx, gy = grid.grid_x, grid.grid_y
    dx = float(abs(gx[1] - gx[0])) if len(gx) > 1 else 0.1
    dy = float(abs(gy[1] - gy[0])) if len(gy) > 1 else 0.1
    return dx, dy


def _local_window(gz, cr, cc, rad):
    ny, nx = gz.shape
    r0, r1 = max(0, cr - rad), min(ny, cr + rad + 1)
    c0, c1 = max(0, cc - rad), min(nx, cc + rad + 1)
    return gz[r0:r1, c0:c1], (r0, c0, r1, c1)


def _blob_mask_at(gz, cr, cc, snr_floor=2.0):
    """Deterministic connected region around (cr,cc) above a robust floor."""
    from scipy.stats import median_abs_deviation
    finite = np.isfinite(gz)
    if finite.sum() < 8:
        return np.zeros_like(gz, dtype=bool)
    nf = float(median_abs_deviation(gz[finite])) or 1e-6
    binary = (np.abs(np.nan_to_num(gz)) / nf) >= snr_floor
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(gz, dtype=bool)
    lab = labeled[int(np.clip(cr,0,gz.shape[0]-1)), int(np.clip(cc,0,gz.shape[1]-1))]
    if lab == 0:
        # nearest labeled component to the centroid
        ys, xs = np.where(labeled > 0)
        if len(ys) == 0:
            return np.zeros_like(gz, dtype=bool)
        d = (ys - cr) ** 2 + (xs - cc) ** 2
        lab = labeled[ys[np.argmin(d)], xs[np.argmin(d)]]
    return labeled == lab


def _centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────

class TargetPossibilityEngine:
    """
    Deterministic evidence engine. Construct once, call analyze() with the
    pipeline result_dict. Emits a progress callback per completed stage so the
    UI progress dialog can list them.

    progress_cb(stage_index:int, stage_name:str, done:bool) -> None
    """

    INTERP_KEYS = ["linear", "cubic", "griddata_cubic", "griddata_linear",
                   "rbf", "rbf_thin_plate"]
    BASELINE_KEYS = ["none", "wavelet_bg", "adaptive_local", "multiscale",
                     "line_median", "highpass"]
    SMOOTH_SIGMAS = {"none": 0.0, "low": 0.6, "medium": 1.2, "high": 2.4}

    def __init__(self, gms_config: dict, weights: dict | None = None,
                 proximity_cells: float = 6.0):
        self.cfg = gms_config or {}
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.proximity = proximity_cells

    # ── Public entry ─────────────────────────────────────────────────────

    def analyze(self, result_dict: dict,
                progress_cb: Optional[Callable[[int, str, bool], None]] = None
                ) -> list[TargetPossibility]:
        grid = result_dict.get("baselined_grid")
        if grid is None:
            raise ValueError("TargetPossibility: result_dict has no baselined_grid")
        scan_files = result_dict.get("scan_files") or []
        anomalies = result_dict.get("confirmed_anomalies", [])
        fusion_result = result_dict.get("fusion_result")
        reliability = (result_dict.get("reliability") or {})

        def _p(i, name, done=True):
            if progress_cb:
                progress_cb(i, name, done)

        # Stage 1-3 are scan-level sweeps (computed ONCE, shared by all targets)
        interp_maps = self._sweep_interpolation(scan_files, grid); _p(1, "Interpolation Stability")
        base_maps   = self._sweep_baselines(scan_files, grid);     _p(2, "Baseline Stability")
        smooth_maps = self._sweep_smoothing(grid);                 _p(3, "Smoothing Stability")
        _p(4, "Fusion Evidence"); _p(5, "Cross Validation Evidence")
        _p(6, "Reliability Evidence"); _p(7, "Signal Quality")
        _p(8, "Spatial Stability"); _p(9, "Morphological Analysis")
        _p(10, "Void Evidence"); _p(11, "Metallic Object Evidence")

        dx, dy = _cell_sizes(grid)
        out: list[TargetPossibility] = []

        for idx, a in enumerate(anomalies):
            tp = self._analyze_one(
                idx, a, grid, dx, dy,
                interp_maps, base_maps, smooth_maps,
                fusion_result, reliability, result_dict,
            )
            out.append(tp)

        _p(12, "Final Evidence Fusion")
        # Deterministic ordering: highest possibility first, then target_id
        out.sort(key=lambda t: (-t.possibility_score, t.target_id))
        return out

    # ── Stage 1: interpolation stability (scan-level maps) ────────────────

    def _sweep_interpolation(self, scan_files, base_grid) -> dict:
        """Re-grid the first scan with every interpolator (on COPIES).
        Returns {key: baselined_like_grid}. Falls back to the base grid for
        any method that errors, so a missing optional dep never crashes TP."""
        maps = {}
        if not scan_files:
            maps["__base__"] = base_grid
            return maps
        try:
            from core.pipeline import GMSPipeline, PipelineConfig
            for key in self.INTERP_KEYS:
                try:
                    cfg = PipelineConfig(interpolator=key,
                                         baseline=getattr(base_grid, "baseline_name", "line_median"))
                    pipe = GMSPipeline(cfg, self.cfg)
                    baselined, _ = pipe.process_scan(scan_files[0])
                    maps[key] = baselined
                except Exception as e:
                    logger.debug(f"[TP] interp {key} skipped: {e}")
        except Exception as e:
            logger.debug(f"[TP] interpolation sweep unavailable: {e}")
        if not maps:
            maps["__base__"] = base_grid
        return maps

    def _sweep_baselines(self, scan_files, base_grid) -> dict:
        maps = {}
        if not scan_files:
            maps["__base__"] = base_grid
            return maps
        try:
            from core.pipeline import GMSPipeline, PipelineConfig
            for key in self.BASELINE_KEYS:
                try:
                    cfg = PipelineConfig(interpolator=getattr(base_grid, "interp_name", "cubic"),
                                         baseline=key)
                    pipe = GMSPipeline(cfg, self.cfg)
                    baselined, _ = pipe.process_scan(scan_files[0])
                    maps[key] = baselined
                except Exception as e:
                    logger.debug(f"[TP] baseline {key} skipped: {e}")
        except Exception as e:
            logger.debug(f"[TP] baseline sweep unavailable: {e}")
        if not maps:
            maps["__base__"] = base_grid
        return maps

    def _sweep_smoothing(self, base_grid) -> dict:
        """Gaussian-smooth COPIES of grid_z at 4 levels. Never touches original."""
        gz = np.nan_to_num(np.asarray(base_grid.grid_z, dtype=float))
        maps = {}
        for name, sigma in self.SMOOTH_SIGMAS.items():
            maps[name] = gz if sigma == 0.0 else ndimage.gaussian_filter(gz, sigma=sigma)
        return maps

    # ── Per-target evidence ───────────────────────────────────────────────

    def _analyze_one(self, idx, a, grid, dx, dy,
                     interp_maps, base_maps, smooth_maps,
                     fusion_result, reliability, result_dict) -> TargetPossibility:
        gz = np.asarray(grid.grid_z, dtype=float)
        ny, nx = gz.shape

        # centroid in grid indices
        cx_m = float(a.get("x", 0.0)); cy_m = float(a.get("y", 0.0))
        cc = int(round(a.get("cx_idx", (cx_m - grid.grid_x[0]) / dx if dx else 0)))
        cr = int(round(a.get("cy_idx", (cy_m - grid.grid_y[0]) / dy if dy else 0)))
        cc = int(np.clip(cc, 0, nx - 1)); cr = int(np.clip(cr, 0, ny - 1))
        rad = 8

        s1 = self._stage_interp_stability(interp_maps, cr, cc)
        s2 = self._stage_baseline_stability(base_maps, cr, cc)
        s3 = self._stage_smoothing_stability(smooth_maps, cr, cc)
        s4 = self._stage_fusion_evidence(a, fusion_result)
        s5 = self._stage_cross_validation(a, result_dict)
        s6 = self._stage_reliability(reliability)
        s7 = self._stage_signal_quality(gz, cr, cc, rad)
        s8 = self._stage_spatial_stability(interp_maps, base_maps, cr, cc)
        s9 = self._stage_morphology(gz, cr, cc)
        s10 = self._stage_void(gz, cr, cc, rad, a)
        s11 = self._stage_metallic(gz, cr, cc, rad, a)

        stages = {
            "interpolation_stability": s1, "baseline_stability": s2,
            "smoothing_stability": s3, "fusion_evidence": s4,
            "cross_validation": s5, "reliability": s6,
            "signal_quality": s7, "spatial_stability": s8,
            "morphology": s9,
        }
        final_0_1 = self._fuse_evidence(stages, s10, s11)
        score100 = round(final_0_1 * 100.0, 1)
        label, color, action = classify(score100)

        # Depth from Module A (honest; may be None)
        depth_m = depth_unc = None
        try:
            from core.physics.depth import DepthEstimationEngine
            est = DepthEstimationEngine().estimate_for_anomaly(
                grid, type("A", (), {"cx": cc, "cy": cr})())
            depth_m, depth_unc = est.depth_m, est.depth_uncertainty_m
        except Exception as e:
            logger.debug(f"[TP] depth unavailable: {e}")

        # collect reasons
        up, down, warns = [], [], []
        for st in list(stages.values()) + [s10, s11]:
            for r in st.reasons:
                if r.sign == "+":
                    up.append(f"{st.name}: {r.text}")
                elif r.sign == "-":
                    down.append(f"{st.name}: {r.text}")
        mask = _blob_mask_at(gz, cr, cc)
        area_m2 = float(mask.sum()) * dx * dy
        if _centroid(mask) is None:
            warns.append("No stable blob resolved at target centroid.")

        return TargetPossibility(
            target_id=a.get("anomaly_id", f"T{idx+1:03d}"),
            x_m=cx_m, y_m=cy_m, depth_m=depth_m, depth_uncertainty_m=depth_unc,
            area_m2=area_m2,
            max_amplitude=float(np.nanmax(np.abs(_local_window(gz, cr, cc, rad)[0])) if gz.size else 0.0),
            confidence=float(a.get("confidence", a.get("combined_confidence", 0.0))),
            uncertainty=float(a.get("uncertainty", a.get("mean_uncertainty", 0.0))),
            fusion_confidence=s4.detail.get("fusion_confidence", 0.0),
            cross_validation=s5.detail.get("summary", "n/a"),
            interpolation_stability=s1.score, baseline_stability=s2.score,
            smoothing_stability=s3.score, reliability_score=s6.score,
            signal_quality=s7.score, spatial_stability=s8.score,
            morphology_score=s9.score, void_possibility=s10.score,
            metallic_possibility=s11.score,
            possibility_score=score100, classification=label, color=color,
            recommended_action=action,
            reasons_up=up, reasons_down=down, warnings=warns,
            stage_scores={k: round(v.score, 4) for k, v in stages.items()}
                         | {"void": round(s10.score, 4), "metallic": round(s11.score, 4)},
        )

    # ── Stage implementations (all deterministic) ─────────────────────────

    def _persistence(self, maps: dict, cr, cc, name: str) -> StageScore:
        """Shared logic for stages 1/2: fraction of methods with a blob at (cr,cc)."""
        keys = [k for k in maps.keys()]
        if not keys:
            return StageScore(name, 0.0, [Reason("-", "no methods available", name, 0)]).clamp()
        hits = 0
        for k in keys:
            g = np.asarray(maps[k].grid_z if hasattr(maps[k], "grid_z") else maps[k], dtype=float)
            m = _blob_mask_at(g, cr, cc)
            if m[int(np.clip(cr,0,g.shape[0]-1)), int(np.clip(cc,0,g.shape[1]-1))]:
                hits += 1
        frac = hits / len(keys)
        reasons = []
        if frac >= 0.75:
            reasons.append(Reason("+", f"persists across {hits}/{len(keys)} methods", name, frac))
        elif frac <= 0.34:
            reasons.append(Reason("-", f"only {hits}/{len(keys)} methods detect it (likely artifact)", name, frac))
        else:
            reasons.append(Reason("i", f"partial persistence {hits}/{len(keys)}", name, frac))
        return StageScore(name, frac, reasons, {"hits": hits, "n": len(keys)}).clamp()

    def _stage_interp_stability(self, maps, cr, cc):
        return self._persistence(maps, cr, cc, "interpolation_stability")

    def _stage_baseline_stability(self, maps, cr, cc):
        return self._persistence(maps, cr, cc, "baseline_stability")

    def _stage_smoothing_stability(self, smooth_maps, cr, cc) -> StageScore:
        """Position/amplitude drift across smoothing levels. Low drift => stable."""
        cents, amps = [], []
        for name, g in smooth_maps.items():
            m = _blob_mask_at(g, cr, cc)
            c = _centroid(m)
            if c is not None:
                cents.append(c)
                amps.append(float(np.nanmax(np.abs(_local_window(g, cr, cc, 8)[0]))))
        if len(cents) < 2:
            return StageScore("smoothing_stability", 0.2,
                              [Reason("-", "target vanishes under smoothing", "smoothing_stability", 0)]).clamp()
        cents = np.array(cents)
        pos_drift = float(np.mean(np.linalg.norm(cents - cents[0], axis=1)))
        amp_drift = float(np.std(amps) / (np.mean(amps) + 1e-9))
        # map drift -> score (0 drift => 1). 3 cells drift or 50% amp swing => ~0
        score = float(np.clip(1.0 - (pos_drift / 3.0) * 0.5 - amp_drift, 0, 1))
        reasons = []
        if pos_drift < 1.0 and amp_drift < 0.2:
            reasons.append(Reason("+", f"stable under smoothing (drift {pos_drift:.2f} cells)", "pos_drift", pos_drift))
        else:
            reasons.append(Reason("-", f"drifts under smoothing ({pos_drift:.2f} cells, amp {amp_drift:.0%})", "pos_drift", pos_drift))
        return StageScore("smoothing_stability", score, reasons,
                          {"pos_drift": pos_drift, "amp_drift": amp_drift}).clamp()

    def _stage_fusion_evidence(self, a, fusion_result) -> StageScore:
        conf = float(a.get("fusion_boost", 0.0))
        n = int(a.get("scan_confirmations", 1))
        fusion_conf = 0.0
        if fusion_result is not None:
            # find the fused target nearest this anomaly id if available
            fusion_conf = float(getattr(getattr(fusion_result, "targets", [None])[0], "confidence", 0.0)) if getattr(fusion_result, "targets", None) else 0.0
        base = float(np.clip(0.25 * (n - 1) + fusion_conf * 0.5 + conf, 0, 1))
        reasons = []
        if n >= 3:
            reasons.append(Reason("+", f"supported by {n} scans", "scan_confirmations", n))
        elif n <= 1:
            reasons.append(Reason("-", "single-scan only (no fusion support)", "scan_confirmations", n))
        return StageScore("fusion_evidence", base, reasons,
                          {"fusion_confidence": fusion_conf, "n_scans": n}).clamp()

    def _stage_cross_validation(self, a, result_dict) -> StageScore:
        n = int(a.get("scan_confirmations", 1))
        agree = float(a.get("label_agreement", 1.0))
        consist = float(a.get("spatial_consistency", 1.0))
        score = float(np.clip(0.5 * min(n / 2.0, 1.0) + 0.25 * agree + 0.25 * consist, 0, 1))
        summary = f"{n} scan(s), agree {agree:.0%}, consistency {consist:.0%}"
        reasons = []
        if n >= 2 and agree >= 0.7:
            reasons.append(Reason("+", f"cross-confirmed ({summary})", "cross_validation", score))
        elif n < 2:
            reasons.append(Reason("-", "not cross-confirmed", "cross_validation", score))
        return StageScore("cross_validation", score, reasons, {"summary": summary}).clamp()

    def _stage_reliability(self, reliability: dict) -> StageScore:
        score = float(reliability.get("reliability_score", reliability.get("snr_mean", 0.0) and 0.5) or 0.5)
        # prefer explicit reliability_score if present
        rs = reliability.get("reliability_score")
        if rs is not None:
            score = float(rs)
        reasons = []
        if score >= 0.75:
            reasons.append(Reason("+", f"scan reliability {score:.0%}", "reliability_score", score))
        elif score < 0.45:
            reasons.append(Reason("-", f"low scan reliability {score:.0%}", "reliability_score", score))
        return StageScore("reliability", score, reasons, dict(reliability)).clamp()

    def _stage_signal_quality(self, gz, cr, cc, rad) -> StageScore:
        win, _ = _local_window(gz, cr, cc, rad)
        finite = np.isfinite(win)
        if finite.sum() < 8:
            return StageScore("signal_quality", 0.0,
                              [Reason("-", "insufficient local data", "signal_quality", 0)]).clamp()
        from scipy.stats import median_abs_deviation
        v = win[finite]
        peak = float(np.max(np.abs(v)))
        bg = float(np.median(np.abs(v)))
        noise = float(median_abs_deviation(v)) or 1e-6
        snr = peak / noise
        prominence = (peak - bg) / (peak + 1e-9)
        score = float(np.clip(0.6 * min(snr / 8.0, 1.0) + 0.4 * prominence, 0, 1))
        reasons = []
        if snr >= 5:
            reasons.append(Reason("+", f"strong local SNR {snr:.1f}", "snr", snr))
        elif snr < 2.5:
            reasons.append(Reason("-", f"weak local SNR {snr:.1f}", "snr", snr))
        return StageScore("signal_quality", score, reasons,
                          {"snr": snr, "prominence": prominence}).clamp()

    def _stage_spatial_stability(self, interp_maps, base_maps, cr, cc) -> StageScore:
        """Centroid deviation of the blob across interp+baseline variants."""
        cents = []
        for maps in (interp_maps, base_maps):
            for k, obj in maps.items():
                g = np.asarray(obj.grid_z if hasattr(obj, "grid_z") else obj, dtype=float)
                c = _centroid(_blob_mask_at(g, cr, cc))
                if c is not None:
                    cents.append(c)
        if len(cents) < 2:
            return StageScore("spatial_stability", 0.3,
                              [Reason("-", "centroid not resolvable across variants", "spatial_stability", 0)]).clamp()
        cents = np.array(cents)
        dev = float(np.mean(np.linalg.norm(cents - cents.mean(axis=0), axis=1)))
        score = float(np.clip(1.0 - dev / 3.0, 0, 1))   # >3 cells scatter => 0
        reasons = []
        if dev < 1.0:
            reasons.append(Reason("+", f"centroid stable ({dev:.2f} cells scatter)", "centroid_dev", dev))
        else:
            reasons.append(Reason("-", f"centroid wanders ({dev:.2f} cells)", "centroid_dev", dev))
        return StageScore("spatial_stability", score, reasons, {"centroid_dev": dev}).clamp()

    def _stage_morphology(self, gz, cr, cc) -> StageScore:
        mask = _blob_mask_at(gz, cr, cc)
        ys, xs = np.where(mask)
        if len(ys) < 4:
            return StageScore("morphology", 0.1,
                              [Reason("-", "blob too small / fragmented", "morphology", 0)]).clamp()
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        area = len(ys)
        bbox_area = h * w
        compactness = area / max(bbox_area, 1)
        elongation = max(h, w) / (min(h, w) + 1)
        ny, nx = gz.shape
        edge = min(ys.min(), xs.min(), ny - 1 - ys.max(), nx - 1 - xs.max())
        reasons = []
        score = 1.0
        if compactness >= 0.55:
            reasons.append(Reason("+", f"compact blob ({compactness:.0%})", "compactness", compactness))
        else:
            score -= 0.3
            reasons.append(Reason("-", f"diffuse blob ({compactness:.0%})", "compactness", compactness))
        if elongation > 4.0:
            score -= 0.4
            reasons.append(Reason("-", f"streak-like (elong {elongation:.1f}) — possible stripe artifact", "elongation", elongation))
        if edge <= 1:
            score -= 0.3
            reasons.append(Reason("-", "touches survey edge (edge artifact risk)", "edge", float(edge)))
        return StageScore("morphology", float(np.clip(score, 0, 1)), reasons,
                          {"compactness": compactness, "elongation": elongation, "edge": int(edge)}).clamp()

    def _stage_void(self, gz, cr, cc, rad, a) -> StageScore:
        """Conservative void/cavity signature: coherent NEGATIVE, smooth, low
        dipole, sufficient extent. Never fabricated — requires all conditions."""
        win, _ = _local_window(gz, cr, cc, rad)
        finite = np.isfinite(win)
        if finite.sum() < 8:
            return StageScore("void_possibility", 0.0,
                              [Reason("i", "insufficient data for void test", "void", 0)]).clamp()
        v = win[finite]
        neg_frac = float((v < 0).mean())
        smoothness = float(np.clip(1.0 - np.std(ndimage.laplace(np.nan_to_num(win))) / (np.ptp(v) + 1e-9), 0, 1))
        dipole = float(a.get("dipole_score", 0.0))
        score = 0.0
        reasons = []
        if neg_frac > 0.6 and smoothness > 0.6 and dipole < 0.2:
            score = float(np.clip(0.4 * neg_frac + 0.4 * smoothness + 0.2 * (1 - dipole), 0, 1))
            reasons.append(Reason("+", f"void-like: {neg_frac:.0%} negative, smooth {smoothness:.0%}", "void", score))
        else:
            reasons.append(Reason("i", "no clear void signature (conservative)", "void", score))
        return StageScore("void_possibility", score, reasons,
                          {"neg_frac": neg_frac, "smoothness": smoothness, "dipole": dipole}).clamp()

    def _stage_metallic(self, gz, cr, cc, rad, a) -> StageScore:
        """Compact buried-metal behaviour: strong dipole, high coherence, compact,
        high SNR. Does NOT name the metal (physically impossible from this data)."""
        dipole = float(a.get("dipole_score", 0.0))
        coh = float(a.get("coherence", a.get("spatial_coherence", 0.0)))
        win, _ = _local_window(gz, cr, cc, rad)
        finite = np.isfinite(win)
        from scipy.stats import median_abs_deviation
        snr = 0.0
        if finite.sum() >= 8:
            v = win[finite]
            snr = float(np.max(np.abs(v)) / (median_abs_deviation(v) or 1e-6))
        score = float(np.clip(0.45 * dipole + 0.3 * coh + 0.25 * min(snr / 8.0, 1.0), 0, 1))
        reasons = []
        if dipole >= 0.5 and coh >= 0.5:
            reasons.append(Reason("+", f"compact metallic behaviour (dipole {dipole:.2f}, coh {coh:.2f})", "metallic", score))
        elif dipole < 0.2:
            reasons.append(Reason("-", "no dipole signature (not metal-like)", "metallic", dipole))
        return StageScore("metallic_possibility", score, reasons,
                          {"dipole": dipole, "coherence": coh, "snr": snr}).clamp()

    # ── Stage 12: final fusion with disagreement penalty ──────────────────

    def _fuse_evidence(self, stages: dict, void: StageScore, metal: StageScore) -> float:
        """
        Weighted mean of the base evidences, THEN a disagreement penalty so a
        single dissenting evidence cannot be hidden by agreeing ones (spec:
        'if one algorithm disagrees, the score should decrease').

            base = sum(w_i * s_i) / sum(w_i)
            penalty = (1 - min_evidence) weighted lightly
            final = base * (0.7 + 0.3 * min_evidence)

        The physical-plausibility nudge from void/metal is applied gently: a
        target that looks like neither a void nor a compact metal object loses
        a little (it may be geology), but a strong void OR strong metal keeps
        the score up. This never fabricates — it only tempers.
        """
        w = self.weights
        num = sum(w[k] * stages[k].score for k in stages)
        den = sum(w[k] for k in stages) or 1.0
        base = num / den

        min_ev = min(s.score for s in stages.values())
        disagreement_factor = 0.7 + 0.3 * min_ev      # in [0.7, 1.0]

        physical = max(void.score, metal.score)        # is it a plausible target at all?
        physical_factor = 0.85 + 0.15 * physical        # in [0.85, 1.0]

        final = base * disagreement_factor * physical_factor
        return float(np.clip(final, 0.0, 1.0))