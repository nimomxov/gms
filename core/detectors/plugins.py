"""
GMS — Anomaly Detector Plugins  v2.0
All detectors implement AnomalyDetectorBase.

Available:
  LoGDetector       — Multi-scale Laplacian of Gaussian (default for cubic)
  AmplitudeDetector — Direct amplitude SNR threshold (required for RBF)
  HybridDetector    — LoG ∪ Amplitude with interior erosion guard

Lesson from v1.4:
  LoG relies on CURVATURE (2nd derivative).
  RBF produces C∞ surfaces → curvature near zero → LoG misses everything.
  Solution: AmplitudeDetector uses |signal|/noise_floor directly.

Adding a new detector:
  1. Subclass AnomalyDetectorBase
  2. Implement name + detect()
  3. Register in REGISTRY
  4. Add to config: detection.mode: "your_name"
"""

import logging
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation

from ..abstractions import (
    AnomalyDetectorBase, BaselinedGrid, DetectionResult,
    RawAnomaly, StageCompatibility
)

logger = logging.getLogger("gms.detector")


# ─────────────────────────────────────────────────────────────────────────────
# Shared feature functions
# ─────────────────────────────────────────────────────────────────────────────

def _robust_snr(region, noise_floor):
    if noise_floor < 1e-6: return 0.0
    return float(np.clip(np.abs(region).max() / noise_floor, 0, 50))

def _smoothness(region):
    lap = ndimage.laplace(region.astype(float))
    rng = float(np.ptp(region))
    if rng < 1e-6: return 1.0
    return float(np.clip(1.0 - np.std(lap) / (rng + 1e-6), 0, 1))

def _dipole_score(region):
    pos = float(region[region > 0].sum()) if (region > 0).any() else 0.0
    neg = float(abs(region[region < 0].sum())) if (region < 0).any() else 0.0
    total = pos + neg
    if total < 1e-6: return 0.0
    return float(np.clip(2.0 * min(pos, neg) / total, 0, 1))

def _polarity(subgrid):
    pos = float(subgrid[subgrid > 0].sum()) if (subgrid > 0).any() else 0.0
    neg = float(abs(subgrid[subgrid < 0].sum())) if (subgrid < 0).any() else 0.0
    return float((pos - neg) / (pos + neg + 1e-6))

def _coherence(blob_mask, gz):
    vals = gz[blob_mask]
    if len(vals) < 3: return 0.0
    dominant = np.sign(np.mean(vals))
    return float(np.sum(np.sign(vals) == dominant) / len(vals))

def _final_score(dipole, polarity, coherence, smoothness):
    pn = float(np.clip((polarity + 1.0) / 2.0, 0, 1))
    return float(np.clip(
        0.35*dipole + 0.25*pn + 0.20*coherence + 0.20*smoothness, 0, 1
    ))

def _uncertainty(snr, smoothness, extent, min_extent):
    return float(np.clip(
        0.5*np.clip(1-snr/10, 0, 1) +
        0.3*np.clip(1-smoothness, 0, 1) +
        0.2*np.clip(1-extent/(min_extent*3), 0, 1),
        0, 1
    ))

def _dipole_midpoint(subgrid, r0, c0):
    if subgrid.max() < 1e-6 or subgrid.min() > -1e-6:
        w = np.abs(subgrid); t = w.sum()
        if t < 1e-6: return float(subgrid.shape[0]//2+r0), float(subgrid.shape[1]//2+c0)
        ri, ci = np.indices(subgrid.shape)
        return float((w*ri).sum()/t+r0), float((w*ci).sum()/t+c0)
    pr, pc = np.unravel_index(np.argmax(subgrid), subgrid.shape)
    nr, nc = np.unravel_index(np.argmin(subgrid), subgrid.shape)
    return float((pr+nr)/2+r0), float((pc+nc)/2+c0)

def _classify(snr, smoothness, dipole, polarity, coherence, fs, uncertainty, rules):
    # NOISE
    if snr < rules.get("NOISE", {}).get("snr_max", 1.8):
        return "NOISE", round(0.25+0.15*(1-uncertainty), 3)
    # FERROUS_METAL
    fm = rules.get("FERROUS_METAL", {})
    if (fs >= fm.get("final_score_min", 0.48) and snr >= fm.get("snr_min", 3.5)
            and coherence >= fm.get("coherence_min", 0.55) and polarity > -0.50):
        conf = 0.45+0.30*min(fs,1)+0.25*min(snr/10,1)
        return "FERROUS_METAL", round(float(np.clip(conf*(1-uncertainty),0,1)),3)
    # CAVITY
    cav = rules.get("CAVITY", {})
    if (fs <= cav.get("final_score_max", 0.42) and snr >= cav.get("snr_min", 3.0)
            and smoothness >= cav.get("smoothness_min", 0.65)):
        conf = 0.38+0.35*smoothness+0.27*min(snr/8,1)
        return "CAVITY", round(float(np.clip(conf*(1-uncertainty),0,1)),3)
    if polarity < -0.75 and snr >= cav.get("snr_min", 3.0) and coherence >= 0.60:
        conf = 0.38+0.35*abs(polarity)+0.27*min(snr/8,1)
        return "CAVITY", round(float(np.clip(conf*(1-uncertainty),0,1)),3)
    # ROCK_DEBRIS
    if snr >= rules.get("ROCK_DEBRIS", {}).get("snr_min", 2.5):
        conf = 0.30+0.30*min(snr/8,1)+0.20*coherence
        return "ROCK_DEBRIS", round(float(np.clip(conf*(1-uncertainty),0,1)),3)
    return "SOIL_VARIATION", round(float(np.clip(0.25*(1-uncertainty),0,1)),3)


def _extract_anomaly(idx, gz, labeled, blob_mask, noise_floor,
                      min_extent, rules, scan_id, detector_name) -> RawAnomaly | None:
    extent = int(blob_mask.sum())
    if extent < min_extent:
        return None

    region = gz[blob_mask]
    rows, cols = np.where(blob_mask)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

    r0 = max(0, bbox[0]-8);  c0 = max(0, bbox[1]-8)
    r1 = min(gz.shape[0], bbox[2]+9); c1 = min(gz.shape[1], bbox[3]+9)
    sub = gz[r0:r1, c0:c1]

    snr  = _robust_snr(region, noise_floor)
    sm   = _smoothness(sub)
    dip  = _dipole_score(sub)
    pol  = _polarity(sub)
    coh  = _coherence(blob_mask, gz)
    fs   = _final_score(dip, pol, coh, sm)
    unc  = _uncertainty(snr, sm, extent, min_extent)
    cy_b = float(np.mean(rows)); cx_b = float(np.mean(cols))

    label, conf = _classify(snr, sm, dip, pol, coh, fs, unc, rules)

    if label == "FERROUS_METAL":
        mcy, mcx = _dipole_midpoint(sub, r0, c0)
    else:
        mcy, mcx = cy_b, cx_b

    return RawAnomaly(
        anomaly_id=f"{scan_id}_A{idx:03d}",
        cx=round(cx_b,2), cy=round(cy_b,2),
        marker_cx=round(mcx,2), marker_cy=round(mcy,2),
        extent_cells=extent,
        peak_amplitude=round(float(np.abs(region).max()),2),
        snr_robust=round(snr,3), smoothness_score=round(sm,3),
        dipole_score=round(dip,3), polarity_ratio=round(pol,3),
        spatial_coherence=round(coh,3), final_score=round(fs,3),
        uncertainty=round(unc,3), raw_label=label, confidence=round(conf,3),
        bbox=bbox, detector_name=detector_name,
    )


def _quality_score(mask, dynamic_range):
    valid_frac = mask.sum() / mask.size
    dr_norm = float(np.clip(dynamic_range / 512.0, 0, 1))
    return round(0.6*valid_frac + 0.4*dr_norm, 3)


# ─────────────────────────────────────────────────────────────────────────────

class LoGDetector(AnomalyDetectorBase):
    """
    Multi-scale Laplacian of Gaussian blob detector.

    Uses second-derivative (curvature) response to find blobs.
    Excellent for noisy/spiky grids (griddata output).

    ⚠️ INCOMPATIBLE with RBF interpolation:
    C∞ smooth surfaces have near-zero curvature → LoG response collapses.
    SNR max typically ~0.3 on RBF grids → nothing detected.

    Parameters:
      snr_min: minimum LoG response / noise_floor to flag a blob
      min_extent: minimum blob size in cells (filters edge noise)
      sigmas: list of Gaussian scales [cells] for multi-scale response
    """

    @property
    def name(self) -> str:
        return "log_detector"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name,
            incompatible_detectors=["rbf_thin_plate", "rbf"],
            notes="⚠️ Fails on RBF grids. Use AmplitudeDetector with RBF."
        )

    def detect(self, grid: BaselinedGrid, config: dict) -> DetectionResult:
        ad = config.get("anomaly_detection", {})
        snr_min    = ad.get("snr_min", 2.6)
        min_extent = ad.get("min_spatial_extent", 5)
        sigmas     = ad.get("multi_scale_sigmas", [1.0, 2.0, 4.0])
        rules      = config.get("classification", {})

        gz, mask = grid.grid_z, grid.grid_mask
        nf = grid.noise_floor

        logger.info(f"[LoG] detecting in {grid.scan_id}")

        responses = [ndimage.gaussian_laplace(gz, s) * s**2 for s in sigmas]
        scale_resp = np.max(np.abs(np.stack(responses)), axis=0) * mask
        snr_grid = np.where(nf > 0, scale_resp / nf, 0)

        labeled, n = ndimage.label(snr_grid > snr_min)
        logger.debug(f"  {n} candidate blobs")

        anomalies = []
        for idx in range(1, n+1):
            a = _extract_anomaly(idx, gz, labeled, labeled==idx,
                                 nf, min_extent, rules, grid.scan_id, self.name)
            if a: anomalies.append(a)

        logger.info(f"  {len(anomalies)} valid anomalies")
        return DetectionResult(
            scan_id=grid.scan_id, anomalies=anomalies,
            scan_quality_score=_quality_score(mask, grid.dynamic_range),
            noise_floor=nf, detector_name=self.name,
            warnings=list(grid.warnings),
        )


class AmplitudeDetector(AnomalyDetectorBase):
    """
    Direct amplitude SNR detector.

    Thresholds on |signal| / noise_floor directly.
    Works on BOTH spiky (griddata) AND ultra-smooth (RBF) grids
    because it does not rely on curvature.

    ⚠️ More prone to edge artifacts on noisy data.
    Mitigated by: mask erosion (removes hull boundary cells) and
    min_spatial_extent (rejects single-cell spikes).

    Parameters:
      snr_min: |signal| / noise_floor threshold
      mask_erosion: number of cells to erode from mask boundary
      min_extent: minimum blob size
    """

    def __init__(self, mask_erosion: int = 3):
        self.mask_erosion = mask_erosion

    @property
    def name(self) -> str:
        return "amplitude_detector"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name,
            notes="Works with both cubic and RBF grids. Required for RBF."
        )

    def detect(self, grid: BaselinedGrid, config: dict) -> DetectionResult:
        ad = config.get("anomaly_detection", {})
        snr_min    = ad.get("snr_min", 2.6)
        min_extent = ad.get("min_spatial_extent", 5)
        rules      = config.get("classification", {})

        gz = grid.grid_z
        nf = grid.noise_floor

        # Erode mask to suppress hull-boundary artifacts
        mask = grid.grid_mask.copy()
        if self.mask_erosion > 0:
            struct = ndimage.generate_binary_structure(2, 1)
            for _ in range(self.mask_erosion):
                mask = ndimage.binary_erosion(mask, structure=struct)

        logger.info(f"[Amplitude] detecting in {grid.scan_id}")

        amp_snr = np.where(nf > 0, np.abs(gz) / nf, 0) * mask
        labeled, n = ndimage.label(amp_snr > snr_min)
        logger.debug(f"  {n} candidate blobs")

        anomalies = []
        for idx in range(1, n+1):
            a = _extract_anomaly(idx, gz, labeled, labeled==idx,
                                 nf, min_extent, rules, grid.scan_id, self.name)
            if a: anomalies.append(a)

        logger.info(f"  {len(anomalies)} valid anomalies")
        return DetectionResult(
            scan_id=grid.scan_id, anomalies=anomalies,
            scan_quality_score=_quality_score(grid.grid_mask, grid.dynamic_range),
            noise_floor=nf, detector_name=self.name,
            warnings=list(grid.warnings),
        )


class HybridDetector(AnomalyDetectorBase):
    """
    LoG ∪ Amplitude union detector.

    Takes the union of LoG blobs and amplitude blobs.
    More sensitive than either alone.
    Uses mask erosion to suppress RBF edge artifacts.

    Best for: griddata cubic with noisy real-world data.
    Overkill for clean synthetic data.
    """

    def __init__(self, mask_erosion: int = 2):
        self.mask_erosion = mask_erosion

    @property
    def name(self) -> str:
        return "hybrid_detector"

    def detect(self, grid: BaselinedGrid, config: dict) -> DetectionResult:
        ad = config.get("anomaly_detection", {})
        snr_min    = ad.get("snr_min", 2.6)
        min_extent = ad.get("min_spatial_extent", 5)
        sigmas     = ad.get("multi_scale_sigmas", [1.0, 2.0, 4.0])
        rules      = config.get("classification", {})

        gz = grid.grid_z
        nf = grid.noise_floor

        mask = grid.grid_mask.copy()
        if self.mask_erosion > 0:
            struct = ndimage.generate_binary_structure(2, 1)
            for _ in range(self.mask_erosion):
                mask = ndimage.binary_erosion(mask, structure=struct)

        logger.info(f"[Hybrid] detecting in {grid.scan_id}")

        responses = [ndimage.gaussian_laplace(gz, s)*s**2 for s in sigmas]
        log_resp  = np.max(np.abs(np.stack(responses)), axis=0) * mask
        log_snr   = np.where(nf > 0, log_resp / nf, 0)
        amp_snr   = np.where(nf > 0, np.abs(gz) / nf, 0) * mask

        binary  = ((log_snr > snr_min) | (amp_snr > snr_min)).astype(int)
        labeled, n = ndimage.label(binary)
        logger.debug(f"  {n} candidate blobs (LoG∪amp)")

        anomalies = []
        for idx in range(1, n+1):
            a = _extract_anomaly(idx, gz, labeled, labeled==idx,
                                 nf, min_extent, rules, grid.scan_id, self.name)
            if a: anomalies.append(a)

        logger.info(f"  {len(anomalies)} valid anomalies")
        return DetectionResult(
            scan_id=grid.scan_id, anomalies=anomalies,
            scan_quality_score=_quality_score(grid.grid_mask, grid.dynamic_range),
            noise_floor=nf, detector_name=self.name,
            warnings=list(grid.warnings),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

from .matched_filter import MatchedDipoleDetector
from .matched_cascade import CascadedMatchedDetector

DETECTOR_REGISTRY: dict[str, type[AnomalyDetectorBase]] = {
    "log":                LoGDetector,
    "log_detector":       LoGDetector,
    "amplitude":          AmplitudeDetector,
    "amplitude_detector": AmplitudeDetector,
    "hybrid":             HybridDetector,
    "hybrid_detector":    HybridDetector,
    "matched_dipole":     MatchedDipoleDetector,
    "matched":            MatchedDipoleDetector,
    "cascaded_matched":   CascadedMatchedDetector,
    "cascade":            CascadedMatchedDetector,
}


def get_detector(mode: str, params: dict = None) -> AnomalyDetectorBase:
    params = params or {}
    if mode not in DETECTOR_REGISTRY:
        raise KeyError(f"Unknown detector '{mode}'. Known: {list(DETECTOR_REGISTRY.keys())}")
    cls = DETECTOR_REGISTRY[mode]
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in params.items() if k in sig.parameters}
    return cls(**valid)
