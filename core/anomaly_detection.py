"""
GMS Core — Anomaly Detection & Feature Extraction  v1.2
Key improvements:
  - Dipole midpoint: ferrous metal marker placed between + and - lobes
  - Weighted FINAL_SCORE = 0.35*dipole + 0.25*polarity_norm + 0.20*coherence + 0.20*smoothness
  - Polarity is a factor, never the sole criterion (guards against mineralization)
"""

import logging
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation
from .signal_processing import ProcessedGrid

logger = logging.getLogger("gms.anomaly")


@dataclass
class Anomaly:
    anomaly_id: str
    cx: float           # blob centroid x (grid index)
    cy: float           # blob centroid y (grid index)
    marker_cx: float    # TRUE marker x — midpoint between lobes for metal
    marker_cy: float    # TRUE marker y — midpoint between lobes for metal
    extent_cells: int
    peak_amplitude: float
    snr_robust: float
    smoothness_score: float
    dipole_score: float
    polarity_ratio: float
    spatial_coherence: float
    final_score: float
    uncertainty: float
    raw_label: str
    confidence: float
    bbox: tuple


@dataclass
class DetectionResult:
    scan_id: str
    anomalies: list[Anomaly] = field(default_factory=list)
    scan_quality_score: float = 0.0
    noise_floor: float = 0.0
    warnings: list = field(default_factory=list)


def _robust_snr(region, noise_floor):
    if noise_floor < 1e-6: return 0.0
    return float(np.clip(np.abs(region).max() / noise_floor, 0, 50))

def _smoothness_score(region):
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

def _polarity_ratio(subgrid):
    pos = float(subgrid[subgrid > 0].sum()) if (subgrid > 0).any() else 0.0
    neg = float(abs(subgrid[subgrid < 0].sum())) if (subgrid < 0).any() else 0.0
    return float((pos - neg) / (pos + neg + 1e-6))

def _spatial_coherence(blob_mask, gz):
    vals = gz[blob_mask]
    if len(vals) < 3: return 0.0
    dominant = np.sign(np.mean(vals))
    return float(np.sum(np.sign(vals) == dominant) / len(vals))

def _compute_final_score(dipole, polarity, coherence, smoothness):
    """
    FINAL_SCORE = 0.35*dipole + 0.25*polarity_norm + 0.20*coherence + 0.20*smoothness
    polarity_norm: [-1,+1] -> [0,1]
    High score -> metal-like. Low score -> cavity/void-like.
    """
    polarity_norm = float(np.clip((polarity + 1.0) / 2.0, 0, 1))
    return float(np.clip(
        0.35 * dipole + 0.25 * polarity_norm + 0.20 * coherence + 0.20 * smoothness,
        0, 1
    ))

def _find_dipole_midpoint(subgrid, r0, c0):
    """
    True object location = midpoint between peak positive and peak negative cell.
    Returns (row_mid, col_mid) in full-grid coordinates.
    Falls back to subgrid centroid if one lobe is absent.
    """
    if subgrid.max() < 1e-6 or subgrid.min() > -1e-6:
        # Only one lobe — use amplitude-weighted centroid of the subgrid
        weights = np.abs(subgrid)
        total = weights.sum()
        if total < 1e-6:
            return float(subgrid.shape[0]//2 + r0), float(subgrid.shape[1]//2 + c0)
        rows_i, cols_i = np.indices(subgrid.shape)
        return float((weights * rows_i).sum()/total + r0), float((weights * cols_i).sum()/total + c0)

    pos_r, pos_c = np.unravel_index(np.argmax(subgrid), subgrid.shape)
    neg_r, neg_c = np.unravel_index(np.argmin(subgrid), subgrid.shape)
    return float((pos_r + neg_r) / 2.0 + r0), float((pos_c + neg_c) / 2.0 + c0)

def _compute_uncertainty(snr, smoothness, extent, min_extent):
    snr_c = float(np.clip(1.0 - snr / 10.0, 0, 1))
    sm_c  = float(np.clip(1.0 - smoothness, 0, 1))
    sz_c  = float(np.clip(1.0 - extent / (min_extent * 3), 0, 1))
    return float(np.clip(0.5*snr_c + 0.3*sm_c + 0.2*sz_c, 0, 1))


class AnomalyDetector:

    def __init__(self, config):
        ad = config.get("anomaly_detection", {})
        self.snr_min    = ad.get("snr_min", 1.8)
        self.min_extent = ad.get("min_spatial_extent", 3)
        self.sigmas     = ad.get("multi_scale_sigmas", [1.0, 2.0, 4.0])
        self.class_rules = config.get("classification", {})

    def detect(self, grid: ProcessedGrid) -> DetectionResult:
        logger.info(f"Detecting anomalies in: {grid.scan_id}")
        warnings = list(grid.warnings)
        gz, mask = grid.grid_z.copy(), grid.grid_mask

        if not mask.any():
            warnings.append("Empty grid mask")
            return DetectionResult(scan_id=grid.scan_id, warnings=warnings)

        # Multi-scale LoG blob detector (primary)
        responses = [ndimage.gaussian_laplace(gz, s) * s**2 for s in self.sigmas]
        scale_resp = np.max(np.abs(np.stack(responses)), axis=0) * mask

        noise_floor = grid.noise_floor
        snr_grid = np.where(noise_floor > 0, scale_resp / noise_floor, 0)
        labeled, n_feat = ndimage.label(snr_grid > self.snr_min)
        logger.debug(f"  {n_feat} candidate blobs")









        anomalies = []
        for idx in range(1, n_feat + 1):
            blob_mask = labeled == idx
            extent = int(blob_mask.sum())
            if extent < self.min_extent:
                continue

            region = gz[blob_mask]
            rows, cols = np.where(blob_mask)
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

            r0 = max(0, bbox[0] - 8);  c0 = max(0, bbox[1] - 8)
            r1 = min(gz.shape[0], bbox[2] + 9);  c1 = min(gz.shape[1], bbox[3] + 9)
            subgrid = gz[r0:r1, c0:c1]

            snr        = _robust_snr(region, noise_floor)
            smoothness = _smoothness_score(subgrid)
            dipole     = _dipole_score(subgrid)
            polarity   = _polarity_ratio(subgrid)
            coherence  = _spatial_coherence(blob_mask, gz)
            fs         = _compute_final_score(dipole, polarity, coherence, smoothness)
            uncertainty = _compute_uncertainty(snr, smoothness, extent, self.min_extent)

            cy_blob = float(np.mean(rows))
            cx_blob = float(np.mean(cols))

            label, confidence = self._classify(snr, smoothness, dipole,
                                                polarity, coherence, fs, uncertainty)

            # Marker placement: metal -> dipole midpoint, others -> centroid
            if label == "FERROUS_METAL":
                marker_cy, marker_cx = _find_dipole_midpoint(subgrid, r0, c0)
            else:
                marker_cy, marker_cx = cy_blob, cx_blob

            anomalies.append(Anomaly(
                anomaly_id=f"{grid.scan_id}_A{idx:03d}",
                cx=round(cx_blob, 2), cy=round(cy_blob, 2),
                marker_cx=round(marker_cx, 2), marker_cy=round(marker_cy, 2),
                extent_cells=extent,
                peak_amplitude=round(float(np.abs(region).max()), 2),
                snr_robust=round(snr, 3),
                smoothness_score=round(smoothness, 3),
                dipole_score=round(dipole, 3),
                polarity_ratio=round(polarity, 3),
                spatial_coherence=round(coherence, 3),
                final_score=round(fs, 3),
                uncertainty=round(uncertainty, 3),
                raw_label=label,
                confidence=round(confidence, 3),
                bbox=bbox,
            ))

        logger.info(f"  {len(anomalies)} valid anomalies")
        valid_frac = mask.sum() / mask.size
        scan_quality = float(0.6*valid_frac + 0.4*np.clip(grid.dynamic_range/512.0, 0, 1))

        return DetectionResult(
            scan_id=grid.scan_id,
            anomalies=anomalies,
            scan_quality_score=round(scan_quality, 3),
            noise_floor=round(noise_floor, 4),
            warnings=warnings,
        )

    def _classify(self, snr, smoothness, dipole, polarity, coherence, final_score, uncertainty):
        rules = self.class_rules

        # NOISE
        if snr < rules.get("NOISE", {}).get("snr_max", 1.8):
            return "NOISE", round(0.25 + 0.15*(1-uncertainty), 3)

        # FERROUS_METAL — requires high final_score AND coherence AND not strongly negative polarity
        fm = rules.get("FERROUS_METAL", {})
        if (final_score >= fm.get("final_score_min", 0.52)
                and snr >= fm.get("snr_min", 3.5)
                and coherence >= fm.get("coherence_min", 0.55)
                and polarity > -0.50):
            conf = 0.45 + 0.30*min(final_score, 1.0) + 0.25*min(snr/10, 1.0)
            return "FERROUS_METAL", round(float(np.clip(conf*(1-uncertainty), 0, 1)), 3)

        # CAVITY — low final_score + smooth + sufficient SNR
        cav = rules.get("CAVITY", {})
        if (final_score <= cav.get("final_score_max", 0.42)
                and snr >= cav.get("snr_min", 3.0)
                and smoothness >= cav.get("smoothness_min", 0.65)):
            conf = 0.38 + 0.35*smoothness + 0.27*min(snr/8, 1.0)
            return "CAVITY", round(float(np.clip(conf*(1-uncertainty), 0, 1)), 3)

        # Dominant-negative polarity fallback (broad suppression)
        if polarity < -0.75 and snr >= cav.get("snr_min", 3.0) and coherence >= 0.60:
            conf = 0.38 + 0.35*abs(polarity) + 0.27*min(snr/8, 1.0)
            return "CAVITY", round(float(np.clip(conf*(1-uncertainty), 0, 1)), 3)

        # ROCK_DEBRIS
        if snr >= rules.get("ROCK_DEBRIS", {}).get("snr_min", 2.5):
            conf = 0.30 + 0.30*min(snr/8, 1.0) + 0.20*coherence
            return "ROCK_DEBRIS", round(float(np.clip(conf*(1-uncertainty), 0, 1)), 3)

        return "SOIL_VARIATION", round(float(np.clip(0.25*(1-uncertainty), 0, 1)), 3)
