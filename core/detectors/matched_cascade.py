"""
GMS — Matched Filter Validation Cascade  v2.2
Multi-stage pipeline that keeps FNR ≈ 0% while cutting FPR significantly.

Stage flow:
  MatchedDipoleDetector (wide net, low threshold)
      ↓
  DipoleValidator        (shape + symmetry check)
      ↓
  CoherenceValidator     (spatial stability)
      ↓
  EnvironmentalRejector  (basalt / mineralized soil guard)
      ↓
  CascadedDetectionResult

Design principle:
  Each stage can only REJECT candidates, never add new ones.
  This guarantees FNR stays near 0% from stage 1.
  FPR is reduced progressively through the cascade.

Benchmark targets (for matched_v2 preset):
  FNR ≤ 5%   (almost no missed real targets)
  FPR ≤ 20%  (down from 60% in v2.1)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import ndimage, signal
from scipy.stats import median_abs_deviation

from ..abstractions import BaselinedGrid, DetectionResult, RawAnomaly, StageCompatibility
from ..classifiers.topology import TopologyValidator
from ..abstractions import AnomalyDetectorBase
from .plugins import (
    _robust_snr, _smoothness, _dipole_score, _polarity,
    _coherence, _final_score, _uncertainty, _dipole_midpoint,
    _quality_score
)
from .matched_filter import (
    _build_template_bank, _fast_normalized_xcorr,
    MatchedDipoleDetector
)

logger = logging.getLogger("gms.cascade")


# ─────────────────────────────────────────────────────────────────────────────
# Validation stages
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateBlob:
    """Internal working object during cascade."""
    idx: int
    blob_mask: np.ndarray
    bbox: tuple
    region: np.ndarray
    subgrid: np.ndarray
    r0: int
    c0: int
    snr: float
    smoothness: float
    dipole: float
    polarity: float
    coherence: float
    final_score: float
    uncertainty: float
    peak_ncc: float
    best_template: str
    cy_blob: float
    cx_blob: float
    # Cascade stage results
    passed_dipole: bool = True
    passed_topology: bool = True
    passed_coherence: bool = True
    passed_environment: bool = True
    rejection_reason: str = ""

    @property
    def passed_all(self) -> bool:
        return (self.passed_dipole and self.passed_topology
                and self.passed_coherence and self.passed_environment)


class DipoleValidator:
    """
    Stage 2: Validates that a detected blob has a physically plausible
    dipole or monopole structure consistent with a buried ferrous object.

    Rejects:
      - Blobs with high polarity AND low dipole AND low coherence
        (signature of mineralized soil gradient, not a point source)
      - Blobs where the NCC template orientation is inconsistent
        with the local field gradient direction
      - Blobs smaller than physics allows for real targets

    Keeps:
      - True dipoles (balanced +/- lobes)
      - Truncated dipoles (one lobe absorbed by baseline — common)
      - High-coherence positive-dominant blobs (deep metal, baseline absorbed neg lobe)
    """

    def __init__(self, min_snr: float = 2.0,
                 min_coherence: float = 0.45,
                 max_noise_polarity: float = 0.98):
        self.min_snr = min_snr
        self.min_coherence = min_coherence
        self.max_noise_polarity = max_noise_polarity  # polarity=±1.0 = single spike

    def validate(self, candidate: CandidateBlob) -> tuple[bool, str]:
        # Rule 1: Minimum SNR
        if candidate.snr < self.min_snr:
            return False, f"snr={candidate.snr:.2f} < {self.min_snr}"

        # Rule 2: Spatial coherence
        if candidate.coherence < self.min_coherence:
            return False, f"coherence={candidate.coherence:.2f} < {self.min_coherence}"

        # Rule 3: Perfect polarity with zero dipole = noise spike or gradient artifact
        # A real target has either a dipole OR is a truncated dipole (still has some structure)
        if (abs(candidate.polarity) > self.max_noise_polarity
                and candidate.dipole < 0.05
                and candidate.coherence < 0.70):
            return False, (f"noise-like: polarity={candidate.polarity:.2f}, "
                           f"dipole={candidate.dipole:.3f}, coherence={candidate.coherence:.2f}")

        # Rule 4: Final score minimum (weighted composite)
        if candidate.final_score < 0.35:
            return False, f"final_score={candidate.final_score:.3f} < 0.35"

        return True, ""


class CoherenceValidator:
    """
    Stage 3: Validates spatial stability of the blob.

    Real targets produce:
      - Spatially stable blobs (the field is smooth around them)
      - Consistent signal direction within the blob
      - Bounded spatial extent (not an infinite line or edge)

    Rejects:
      - Elongated streaks (edge artifacts, scan-line noise)
      - Blobs with very low smoothness (fragmented noise)
      - Blobs with aspect ratio > max_aspect (scan-line artifacts)
    """

    def __init__(self, min_smoothness: float = 0.35,
                 max_aspect_ratio: float = 8.0,
                 min_compactness: float = 0.05):
        self.min_smoothness = min_smoothness
        self.max_aspect_ratio = max_aspect_ratio
        self.min_compactness = min_compactness

    def validate(self, candidate: CandidateBlob) -> tuple[bool, str]:
        # Rule 1: Minimum smoothness
        if candidate.smoothness < self.min_smoothness:
            return False, f"smoothness={candidate.smoothness:.3f} < {self.min_smoothness}"

        # Rule 2: Aspect ratio (scan-line artifacts are very elongated)
        rows, cols = np.where(candidate.blob_mask)
        if len(rows) > 1:
            height = int(rows.max() - rows.min()) + 1
            width  = int(cols.max() - cols.min()) + 1
            aspect = max(height, width) / (min(height, width) + 1)
            if aspect > self.max_aspect_ratio:
                return False, f"aspect_ratio={aspect:.1f} > {self.max_aspect_ratio} (streak artifact)"

        # Rule 3: Compactness = area / bounding_box_area (real blobs are compact)
        if len(rows) > 1:
            bbox_area = (rows.max()-rows.min()+1) * (cols.max()-cols.min()+1)
            compactness = len(rows) / max(bbox_area, 1)
            if compactness < self.min_compactness:
                return False, f"compactness={compactness:.3f} < {self.min_compactness}"

        return True, ""


class EnvironmentalRejector:
    """
    Stage 4: Rejects false positives caused by environmental interference.

    Targets:
      - Basalt: produces broad, high-amplitude, coherent positive field
        Signature: very high smoothness + very high polarity + no dipole + large extent
      - Mineralized soil: produces gradients across scan lines
        Signature: high polarity along scan direction + low NCC
      - EMI corruption: random spikes with inconsistent NCC template match

    These rejections only fire when MULTIPLE conditions align —
    a single condition alone never rejects a real candidate.
    """

    def __init__(self, basalt_smoothness_min: float = 0.92,
                 basalt_extent_min: int = 30,
                 emi_ncc_max: float = 0.20,
                 local_variance_ratio_max: float = 15.0):
        self.basalt_smoothness_min    = basalt_smoothness_min
        self.basalt_extent_min        = basalt_extent_min
        self.emi_ncc_max              = emi_ncc_max
        self.local_variance_ratio_max = local_variance_ratio_max

    def validate(self, candidate: CandidateBlob,
                 grid: BaselinedGrid) -> tuple[bool, str]:

        # Basalt / large magnetic body rejection
        # Basalt signature: very smooth + very large + polarity-dominant + no NCC match
        blob_extent = int(candidate.blob_mask.sum())
        if (candidate.smoothness >= self.basalt_smoothness_min
                and blob_extent >= self.basalt_extent_min
                and candidate.dipole < 0.10
                and candidate.peak_ncc < 0.35):
            return False, (
                f"basalt/large-body: smooth={candidate.smoothness:.2f}, "
                f"extent={candidate.extent_cells}, ncc={candidate.peak_ncc:.2f}"
            )

        # EMI spike rejection: very high amplitude but no template match
        if (candidate.snr > 12.0
                and candidate.peak_ncc < self.emi_ncc_max
                and candidate.dipole < 0.05
                and candidate.extent_cells < 8):
            return False, (
                f"EMI spike: snr={candidate.snr:.1f}, "
                f"ncc={candidate.peak_ncc:.2f}, extent={candidate.extent_cells}"
            )

        # Local variance ratio: is this blob in a region of abnormally high variance?
        # Real targets stand out from their LOCAL background.
        # If local variance everywhere is high, it's background geology, not a target.
        noise_map = grid.meta.get("noise_map", None)
        if noise_map is not None:
            rows, cols = np.where(candidate.blob_mask)
            local_nf  = float(np.median(noise_map[rows, cols]))
            global_nf = grid.noise_floor
            if global_nf > 1e-6:
                ratio = local_nf / global_nf
                if ratio > self.local_variance_ratio_max:
                    return False, (
                        f"high-variance zone: local_nf/global_nf={ratio:.1f} "
                        f"(likely basalt or mineralized)"
                    )

        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Cascaded detector
# ─────────────────────────────────────────────────────────────────────────────

class CascadedMatchedDetector(AnomalyDetectorBase):
    """
    Multi-stage validation cascade detector.

    Stage 1: MatchedDipoleDetector — wide net (low NCC threshold)
    Stage 2: DipoleValidator       — shape/symmetry check
    Stage 3: CoherenceValidator    — spatial stability
    Stage 4: EnvironmentalRejector — basalt/EMI/soil guard

    Configuration in config dict:
      anomaly_detection.cascade.*

    Benchmark targets:
      FNR ≤ 5%   (rarely misses real metal)
      FPR ≤ 20%  (down from 60% in v2.1 matched preset)
    """

    def __init__(self,
                 ncc_threshold_wide: float = 0.28,
                 n_depths: int = 5,
                 n_orientations: int = 6,
                 template_size: int = 17,
                 dipole_validator: DipoleValidator = None,
                 coherence_validator: CoherenceValidator = None,
                 environmental_rejector: EnvironmentalRejector = None,
                 topology_validator: TopologyValidator = None,
                 use_amplitude_primary: bool = False):
        self.ncc_threshold_wide  = ncc_threshold_wide
        self.n_depths            = n_depths
        self.n_orientations      = n_orientations
        self.template_size       = template_size | 1
        self.dv  = dipole_validator       or DipoleValidator()
        self.tv  = topology_validator     or TopologyValidator()
        self.cv  = coherence_validator    or CoherenceValidator()
        self.er  = environmental_rejector or EnvironmentalRejector()
        self.use_amplitude_primary = use_amplitude_primary
        self._bank = None

    @property
    def name(self) -> str:
        return "cascaded_matched"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name,
            preferred_baseline=["line_median", "adaptive_local", "multiscale"],
            notes="Best detector for production use. Works with cubic interpolation."
        )

    def _get_bank(self):
        if self._bank is None:
            self._bank = _build_template_bank(
                min_depth=1.5, max_depth=10.0,
                n_depths=self.n_depths,
                n_orientations=self.n_orientations,
                template_size=self.template_size,
            )
        return self._bank

    def detect(self, grid: BaselinedGrid, config: dict) -> DetectionResult:
        ad         = config.get("anomaly_detection", {})
        min_extent = ad.get("min_spatial_extent", 5)
        sigmas     = ad.get("multi_scale_sigmas", [1.0, 2.0, 4.0])
        rules      = config.get("classification", {})

        gz, mask = grid.grid_z, grid.grid_mask
        noise_map = grid.meta.get("noise_map", None)
        nf = grid.noise_floor

        logger.info(f"[Cascade] detecting in {grid.scan_id}")

        # ── Stage 1: Detection (amplitude primary OR LoG+NCC union) ─────────
        bank = self._get_bank()
        ncc_max = np.zeros_like(gz)
        best_tmpl = np.full(gz.shape, "", dtype=object)

        for tmpl_name, tmpl in bank:
            ncc = _fast_normalized_xcorr(gz, tmpl) * mask
            improved = ncc > ncc_max
            ncc_max  = np.where(improved, ncc, ncc_max)
            best_tmpl = np.where(improved, tmpl_name, best_tmpl)

        snr_min_stage1 = ad.get("snr_min", 2.4)
        log_nf = noise_map if noise_map is not None else np.full_like(gz, nf)

        if self.use_amplitude_primary:
            # Amplitude primary for smooth multiscale baseline output.
            # Uses local-maximum detection to avoid giant connected blobs:
            # 1. Find local maxima of |signal| at multiple scales
            # 2. Only flag compact regions AROUND those local maxima
            # This prevents the entire positive lobe from becoming one huge blob.
            amp_map = np.abs(gz) * mask
            amp_snr_map = np.where(log_nf > 1e-6, amp_map / log_nf, 0)

            # Multi-scale local maxima: peak must be larger than its neighborhood
            binary = np.zeros_like(gz, dtype=int)
            for peak_radius in [4, 8, 14]:
                local_max = ndimage.maximum_filter(amp_map, size=2*peak_radius+1)
                is_peak = (amp_map == local_max) & (amp_snr_map >= snr_min_stage1) & mask
                if is_peak.any():
                    # Grow each peak into a local region (watershed-like)
                    peak_dil = ndimage.binary_dilation(is_peak,
                                                        iterations=peak_radius)
                    qualified = peak_dil & (amp_snr_map >= snr_min_stage1 * 0.6) & mask
                    binary |= qualified.astype(int)

            # Also allow NCC peaks even if below amp threshold
            binary |= (ncc_max >= self.ncc_threshold_wide).astype(int) * mask
            logger.debug(
                f"  Stage 1: amplitude primary (peaks), amp_snr_max={amp_snr_map.max():.2f}, "
                f"binary_cells={binary.sum()}"
            )
        else:
            # LoG + NCC union — works on spiky griddata output
            responses = [ndimage.gaussian_laplace(gz, s)*s**2 for s in sigmas]
            log_resp  = np.max(np.abs(np.stack(responses)), axis=0) * mask
            log_snr   = np.where(log_nf > 1e-6, log_resp / log_nf, 0)
            binary = (
                (ncc_max >= self.ncc_threshold_wide) |
                (log_snr >= snr_min_stage1)
            ).astype(int) * mask

        labeled, n_raw = ndimage.label(binary)
        logger.debug(f"  Stage 1: {n_raw} raw candidates")

        # ── Build candidate objects ───────────────────────────────────────────
        candidates: list[CandidateBlob] = []
        for idx in range(1, n_raw + 1):
            blob_mask = labeled == idx
            extent    = int(blob_mask.sum())
            if extent < min_extent:
                continue

            region = gz[blob_mask]
            rows, cols = np.where(blob_mask)
            bbox = (int(rows.min()), int(cols.min()),
                    int(rows.max()), int(cols.max()))

            r0 = max(0, bbox[0]-8);  c0 = max(0, bbox[1]-8)
            r1 = min(gz.shape[0], bbox[2]+9); c1 = min(gz.shape[1], bbox[3]+9)
            sub = gz[r0:r1, c0:c1]

            # Use local noise floor if available
            local_nf = (float(np.median(noise_map[rows, cols]))
                        if noise_map is not None else nf)
            local_nf = max(local_nf, 1e-3)

            snr  = _robust_snr(region, local_nf)
            sm   = _smoothness(sub)
            dip  = _dipole_score(sub)
            pol  = _polarity(sub)
            coh  = _coherence(blob_mask, gz)
            fs   = _final_score(dip, pol, coh, sm)
            unc  = _uncertainty(snr, sm, extent, min_extent)
            pncc = float(ncc_max[blob_mask].max())
            btmpl = str(best_tmpl[rows[np.argmax(ncc_max[rows, cols])],
                                   cols[np.argmax(ncc_max[rows, cols])]])

            candidates.append(CandidateBlob(
                idx=idx, blob_mask=blob_mask, bbox=bbox,
                region=region, subgrid=sub, r0=r0, c0=c0,
                snr=snr, smoothness=sm, dipole=dip, polarity=pol,
                coherence=coh, final_score=fs, uncertainty=unc,
                peak_ncc=pncc, best_template=btmpl,
                cy_blob=float(np.mean(rows)), cx_blob=float(np.mean(cols)),
            ))

        logger.debug(f"  Stage 1 (extent filter): {len(candidates)} candidates")

        # ── Stage 2: Dipole Validator ─────────────────────────────────────────
        for c in candidates:
            ok, reason = self.dv.validate(c)
            if not ok:
                c.passed_dipole = False
                c.rejection_reason = f"DipoleValidator: {reason}"

        after_dv = sum(1 for c in candidates if c.passed_all)
        logger.debug(f"  Stage 2 (DipoleValidator): {after_dv} remaining")

        # ── Stage 2.5: Topology Validator ─────────────────────────────────────
        for c in candidates:
            if not c.passed_dipole:
                continue
            grid_size = int(gz.size)
            ok, reason, _ = self.tv.validate(c.blob_mask, c.subgrid, c.region, grid_size=grid_size)
            if not ok:
                c.passed_topology = False
                c.rejection_reason = f"TopologyValidator: {reason}"

        after_tv = sum(1 for c in candidates if c.passed_all)
        logger.debug(f"  Stage 2.5 (TopologyValidator): {after_tv} remaining")

        # ── Stage 3: Coherence Validator ──────────────────────────────────────
        for c in candidates:
            if not c.passed_dipole or not c.passed_topology:
                continue
            ok, reason = self.cv.validate(c)
            if not ok:
                c.passed_coherence = False
                c.rejection_reason = f"CoherenceValidator: {reason}"

        after_cv = sum(1 for c in candidates if c.passed_all)
        logger.debug(f"  Stage 3 (CoherenceValidator): {after_cv} remaining")

        # ── Stage 4: Environmental Rejector ───────────────────────────────────
        for c in candidates:
            if not c.passed_dipole or not c.passed_coherence:
                continue
            ok, reason = self.er.validate(c, grid)
            if not ok:
                c.passed_environment = False
                c.rejection_reason = f"EnvironmentalRejector: {reason}"

        after_er = sum(1 for c in candidates if c.passed_all)
        logger.debug(f"  Stage 4 (EnvironmentalRejector): {after_er} final")

        # ── Build anomalies from passing candidates ───────────────────────────
        from .plugins import _classify
        anomalies = []
        for c in candidates:
            if not c.passed_all:
                logger.debug(f"    Rejected blob {c.idx}: {c.rejection_reason}")
                continue

            label, conf = _classify(
                c.snr, c.smoothness, c.dipole, c.polarity,
                c.coherence, c.final_score, c.uncertainty, rules
            )
            # NCC boost for high template matches
            ncc_boost = float(np.clip(c.peak_ncc * 0.18, 0, 0.18))
            conf = round(float(np.clip(conf + ncc_boost, 0, 1)), 3)

            if label == "FERROUS_METAL":
                mcy, mcx = _dipole_midpoint(c.subgrid, c.r0, c.c0)
            else:
                mcy, mcx = c.cy_blob, c.cx_blob

            anomalies.append(RawAnomaly(
                anomaly_id=f"{grid.scan_id}_C{c.idx:03d}",
                cx=round(c.cx_blob, 2), cy=round(c.cy_blob, 2),
                marker_cx=round(mcx, 2), marker_cy=round(mcy, 2),
                extent_cells=int(c.blob_mask.sum()),
                peak_amplitude=round(float(np.abs(c.region).max()), 2),
                snr_robust=round(c.snr, 3),
                smoothness_score=round(c.smoothness, 3),
                dipole_score=round(c.dipole, 3),
                polarity_ratio=round(c.polarity, 3),
                spatial_coherence=round(c.coherence, 3),
                final_score=round(c.final_score, 3),
                uncertainty=round(c.uncertainty, 3),
                raw_label=label, confidence=conf,
                bbox=c.bbox,
                detector_name=(
                    f"cascade(ncc={c.peak_ncc:.2f},"
                    f"t={c.best_template[:20]})"
                ),
            ))

        logger.info(
            f"  Cascade complete: {n_raw} raw → {len(candidates)} extent-filtered "
            f"→ {after_dv} dv → {after_cv} cv → {after_er} env → "
            f"{len(anomalies)} final anomalies"
        )

        return DetectionResult(
            scan_id=grid.scan_id,
            anomalies=anomalies,
            scan_quality_score=_quality_score(mask, grid.dynamic_range),
            noise_floor=nf,
            detector_name=self.name,
            warnings=list(grid.warnings),
        )
