"""
GMS — Global Reliability Engine  v2.3

Problem solved:
  Even when a local blob looks like a dipole, if the ENTIRE SCAN
  is noisy/unstable/incoherent, DIG confidence should be penalized.
  This stops noise_only → RESCAN and high_noise → DIG false calls.

Design:
  ScanReliabilityEngine assesses the global quality of a processed scan
  and returns a ScanReliability dataclass with:
    - reliability_score  [0, 1]   overall scan usability
    - penalty_factor     [0, 1]   multiplier applied to anomaly confidence
    - flags              dict     which specific quality issues were detected

  The pipeline multiplies anomaly.confidence × penalty_factor.
  If penalty_factor is low, DIG → RESCAN automatically.

Reliability components:
  1. SNR_global     — median signal-to-noise across the valid grid
  2. Coherence_global — spatial autocorrelation of signal (random noise has low autocorr)
  3. Stability_score  — variance of the noise floor across scan rows (unstable scan)
  4. Coverage_fraction — what fraction of the grid has valid data
  5. Baseline_residual — how much energy remained after baseline removal (drift quality)

The final penalty_factor is the geometric mean of all components.
Geometric mean is preferred: a single very bad component → strong penalty.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation

from .abstractions import BaselinedGrid

logger = logging.getLogger("gms.reliability")


@dataclass
class ScanReliability:
    """Per-scan reliability assessment."""
    scan_id: str
    reliability_score: float     # [0,1] overall quality
    penalty_factor: float        # [0,1] multiply onto anomaly confidence
    snr_global: float
    coherence_global: float
    stability_score: float
    coverage_fraction: float
    baseline_residual: float
    flags: dict = field(default_factory=dict)
    message: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.reliability_score >= 0.45

    @property
    def quality_label(self) -> str:
        if self.reliability_score >= 0.75: return "GOOD"
        if self.reliability_score >= 0.50: return "MARGINAL"
        if self.reliability_score >= 0.30: return "POOR"
        return "UNRELIABLE"


class ScanReliabilityEngine:
    """
    Assesses global scan quality and produces a reliability penalty factor.

    Usage in pipeline:
        rel = engine.assess(baselined_grid)
        for anomaly in detection.anomalies:
            anomaly.confidence = min(anomaly.confidence, rel.penalty_factor * anomaly.confidence)
    """

    def __init__(self,
                 snr_min_reliable: float = 3.0,
                 snr_max_reliable: float = 35.0,
                 coherence_window: int = 5,
                 stability_max_cv: float = 0.80,
                 min_coverage: float = 0.50):
        self.snr_min_reliable    = snr_min_reliable
        self.snr_max_reliable    = snr_max_reliable
        self.coherence_window    = coherence_window
        self.stability_max_cv    = stability_max_cv
        self.min_coverage        = min_coverage

    def assess(self, grid: BaselinedGrid) -> ScanReliability:
        gz   = grid.grid_z
        mask = grid.grid_mask
        nf   = grid.noise_floor
        flags = {}

        # ── 1. Global SNR ─────────────────────────────────────────────────────
        valid_vals = gz[mask]
        if valid_vals.size < 4:
            return ScanReliability(
                scan_id=grid.scan_id, reliability_score=0.0, penalty_factor=0.0,
                snr_global=0, coherence_global=0, stability_score=0,
                coverage_fraction=0, baseline_residual=0,
                flags={"empty_grid": True},
                message="Empty or near-empty grid — scan unusable",
            )

        signal_rms = float(np.sqrt(np.mean(valid_vals**2)))
        snr_global = float(np.clip(signal_rms / (nf + 1e-6), 0, 50))

        # SNR component: penalize both too-low (noisy) and too-high (saturated)
        if snr_global < self.snr_min_reliable:
            snr_component = float(np.clip(snr_global / self.snr_min_reliable, 0, 1))
            flags["low_snr"] = f"{snr_global:.2f} < {self.snr_min_reliable}"
        elif snr_global > self.snr_max_reliable:
            snr_component = 0.5  # partial penalty for saturation
            flags["high_snr_saturation"] = f"{snr_global:.1f}"
        else:
            snr_component = 1.0

        # ── 2. Spatial Coherence (autocorrelation proxy) ──────────────────────
        # Random noise has near-zero spatial autocorrelation.
        # Real signal has structure → positive autocorrelation.
        coherence_global = _spatial_autocorr(gz, mask, lag=self.coherence_window)
        coherence_component = float(np.clip(coherence_global / 0.30, 0, 1))
        if coherence_global < 0.10:
            flags["low_spatial_coherence"] = f"{coherence_global:.3f}"

        # ── 3. Row-to-row stability (noise floor variance) ────────────────────
        # A stable scan has consistent noise floor across rows.
        # Unstable = large row-to-row MAD variation (scanner instability/EMI).
        row_nf = []
        for r in range(gz.shape[0]):
            v = gz[r][mask[r]]
            if len(v) > 4:
                row_nf.append(float(median_abs_deviation(v)))
        if row_nf:
            nf_mean = float(np.mean(row_nf))
            nf_std  = float(np.std(row_nf))
            cv = nf_std / (nf_mean + 1e-6)   # coefficient of variation
            stability_score = float(np.clip(1.0 - cv / self.stability_max_cv, 0, 1))
            if cv > self.stability_max_cv * 0.6:
                flags["unstable_noise_floor"] = f"CV={cv:.2f}"
        else:
            stability_score = 0.5

        # ── 4. Coverage ───────────────────────────────────────────────────────
        coverage = float(mask.sum() / mask.size)
        coverage_component = float(np.clip(coverage / self.min_coverage, 0, 1))
        if coverage < self.min_coverage:
            flags["low_coverage"] = f"{coverage:.0%}"

        # ── 5. Baseline residual quality ──────────────────────────────────────
        # After baseline, ideal residual has small mean (DC ≈ 0).
        # Large mean → baseline didn't fully remove drift.
        residual_mean = float(abs(np.mean(valid_vals)))
        residual_ratio = residual_mean / (nf + 1e-6)
        baseline_residual = float(np.clip(1.0 - residual_ratio / 5.0, 0, 1))
        if residual_ratio > 3.0:
            flags["poor_baseline"] = f"mean/nf={residual_ratio:.1f}"

        # ── Geometric mean → overall reliability ──────────────────────────────
        components = [snr_component, coherence_component, stability_score,
                      coverage_component, baseline_residual]
        # Geometric mean: any single component near 0 → strong penalty
        reliability_score = float(np.prod([max(c, 1e-4) for c in components]) ** (1/len(components)))
        reliability_score = float(np.clip(reliability_score, 0, 1))

        # Penalty factor: more lenient than reliability (don't over-penalize)
        # Map [0,1] → [0.3, 1.0] so even poor scans retain 30% confidence
        penalty_factor = float(np.clip(0.30 + 0.70 * reliability_score, 0, 1))

        message = (
            f"quality={self.quality_label_from(reliability_score)}  "
            f"snr={snr_global:.1f}  coh={coherence_global:.3f}  "
            f"stab={stability_score:.2f}  cov={coverage:.0%}"
        )
        if flags:
            message += f"  flags={list(flags.keys())}"

        rel = ScanReliability(
            scan_id=grid.scan_id,
            reliability_score=round(reliability_score, 3),
            penalty_factor=round(penalty_factor, 3),
            snr_global=round(snr_global, 3),
            coherence_global=round(coherence_global, 3),
            stability_score=round(stability_score, 3),
            coverage_fraction=round(coverage, 3),
            baseline_residual=round(baseline_residual, 3),
            flags=flags,
            message=message,
        )

        logger.info(f"  Reliability [{grid.scan_id[:16]}]: {message}")
        return rel

    @staticmethod
    def quality_label_from(score: float) -> str:
        if score >= 0.75: return "GOOD"
        if score >= 0.50: return "MARGINAL"
        if score >= 0.30: return "POOR"
        return "UNRELIABLE"

    def apply_penalty(self, anomalies: list, reliability: ScanReliability) -> list:
        """
        Apply reliability penalty to anomaly confidences.
        If reliability is poor, downgrade confidences proportionally.
        """
        pf = reliability.penalty_factor
        if pf >= 0.95:
            return anomalies  # no penalty needed

        for a in anomalies:
            a_dict = a.__dict__
            old_conf = a_dict.get("confidence", 0)
            new_conf = round(float(np.clip(old_conf * pf, 0, 1)), 3)
            a.__dict__["confidence"] = new_conf
            if old_conf > new_conf:
                logger.debug(
                    f"  Reliability penalty: {a.anomaly_id} "
                    f"conf {old_conf:.3f}→{new_conf:.3f} (pf={pf:.2f})"
                )
        return anomalies


def _spatial_autocorr(gz: np.ndarray, mask: np.ndarray, lag: int = 5) -> float:
    """
    Moran's I approximation — spatial autocorrelation at given lag.
    Range: [-1, 1]. Pure random noise → 0. Smooth signal → positive.
    """
    try:
        valid = gz * mask
        mean_val = float(np.mean(valid[mask])) if mask.any() else 0.0
        centered = (valid - mean_val) * mask

        # Horizontal lag-correlation
        if gz.shape[1] > lag + 1:
            c_h = float(np.sum(centered[:, :-lag] * centered[:, lag:] * mask[:, :-lag] * mask[:, lag:]))
            n_h = float(np.sum(mask[:, :-lag] * mask[:, lag:]))
        else:
            c_h, n_h = 0.0, 1.0

        # Vertical lag-correlation
        if gz.shape[0] > lag + 1:
            c_v = float(np.sum(centered[:-lag, :] * centered[lag:, :] * mask[:-lag, :] * mask[lag:, :]))
            n_v = float(np.sum(mask[:-lag, :] * mask[lag:, :]))
        else:
            c_v, n_v = 0.0, 1.0

        numerator   = (c_h / (n_h + 1e-8) + c_v / (n_v + 1e-8)) / 2
        denominator = float(np.mean(centered[mask]**2)) if mask.any() else 1.0
        return float(np.clip(numerator / (denominator + 1e-8), -1, 1))
    except Exception:
        return 0.0
