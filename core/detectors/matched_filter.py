"""
GMS — Matched Dipole Filter Detector
Cross-correlates the grid with a synthetic magnetic dipole template.

WHY this is better than LoG:
  LoG detects any blob with curvature. It is generic.
  A matched filter detects SPECIFICALLY the expected shape of a
  ferrous magnetic dipole. It ignores blobs that don't match the
  dipole pattern — massive reduction in false positives.

Physics of a vertical magnetic dipole (simplified 2D):
  V(x,y) = A * (2z²-r²) / (r²+z²)^(5/2)
  where r = sqrt(x²+y²), z = depth (abstract), A = amplitude.

  This produces:
    - Central positive lobe (above object)
    - Annular negative region (surrounding the object)

For a gradiometer at angle θ (magnetic inclination), the pattern is
asymmetric (positive lobe shifts toward magnetic equator/pole).
Here we use a simplified vertical dipole template (inclination = 90°)
plus a horizontal variant for robustness.

Detection method:
  1. Build template bank (multiple scales + orientations)
  2. Normalized cross-correlation with each template
  3. Take max response across templates
  4. Threshold on correlation coefficient (range [-1, 1])

Output: correlation map where high values = strong dipole match.
"""

import logging
import numpy as np
from scipy import ndimage, signal
from scipy.stats import median_abs_deviation

from ..abstractions import (
    AnomalyDetectorBase, BaselinedGrid, DetectionResult, StageCompatibility
)
from .plugins import (
    _extract_anomaly, _quality_score, _classify,
    _robust_snr, _smoothness, _dipole_score, _polarity,
    _coherence, _final_score, _uncertainty, _dipole_midpoint
)
from ..abstractions import RawAnomaly

logger = logging.getLogger("gms.matched_filter")


# ─────────────────────────────────────────────────────────────────────────────
# Template generation
# ─────────────────────────────────────────────────────────────────────────────

def _vertical_dipole_template(size: int, depth_cells: float = 3.0) -> np.ndarray:
    """
    2D vertical magnetic dipole template.

    V(x,y) ∝ (2z² - x² - y²) / (x² + y² + z²)^(5/2)

    Central positive lobe + surrounding negative annulus.
    'depth_cells' controls the compactness: larger depth → broader template.
    """
    half = size // 2
    y_idx, x_idx = np.mgrid[-half:half+1, -half:half+1]
    r2 = x_idx**2 + y_idx**2
    z2 = depth_cells**2
    template = (2*z2 - r2) / (r2 + z2)**2.5
    # Normalize to zero mean, unit norm
    template -= template.mean()
    norm = np.linalg.norm(template)
    if norm > 1e-10:
        template /= norm
    return template


def _horizontal_dipole_template(size: int, depth_cells: float = 3.0,
                                  angle_deg: float = 0.0) -> np.ndarray:
    """
    2D horizontal dipole template at angle_deg from north.
    Used for non-vertical magnetization (tilted objects, equatorial areas).

    The horizontal dipole has an antisymmetric pattern:
      positive lobe on one side, negative on the other.
    """
    half = size // 2
    y_idx, x_idx = np.mgrid[-half:half+1, -half:half+1]
    angle_rad = np.deg2rad(angle_deg)
    # Project onto dipole axis
    axis_x = np.cos(angle_rad)
    axis_y = np.sin(angle_rad)
    dx = x_idx * axis_x + y_idx * axis_y
    r2 = x_idx**2 + y_idx**2
    z2 = depth_cells**2
    template = dx / (r2 + z2)**2.0
    template -= template.mean()
    norm = np.linalg.norm(template)
    if norm > 1e-10:
        template /= norm
    return template


def _build_template_bank(min_depth: float = 2.0,
                          max_depth: float = 8.0,
                          n_depths: int = 4,
                          n_orientations: int = 4,
                          template_size: int = 21) -> list[tuple[str, np.ndarray]]:
    """
    Build a bank of dipole templates at multiple scales and orientations.
    Returns list of (label, template_array).
    """
    bank = []
    depths = np.linspace(min_depth, max_depth, n_depths)

    for d in depths:
        # Vertical dipole (most common in mid-latitudes)
        tmpl = _vertical_dipole_template(template_size, depth_cells=d)
        bank.append((f"vertical_d{d:.1f}", tmpl))

        # Horizontal dipoles at multiple angles (for tilted/equatorial targets)
        for angle in np.linspace(0, 180, n_orientations, endpoint=False):
            tmpl_h = _horizontal_dipole_template(template_size, d, angle)
            bank.append((f"horizontal_d{d:.1f}_a{angle:.0f}", tmpl_h))

    logger.debug(f"  Template bank: {len(bank)} templates "
                 f"({n_depths} depths × {1+n_orientations} orientations)")
    return bank


# ─────────────────────────────────────────────────────────────────────────────
# Normalized cross-correlation
# ─────────────────────────────────────────────────────────────────────────────

def _normalized_xcorr(grid: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Normalized cross-correlation (NCC) between grid and template.
    Output range: [-1, +1]. High positive value = strong template match.

    Uses sliding window normalization to handle non-uniform backgrounds.
    """
    t_size = template.shape[0]
    half   = t_size // 2

    # Pad grid to handle boundaries
    gz_pad = np.pad(grid, half, mode="reflect")

    nrows, ncols = grid.shape
    ncc = np.zeros_like(grid)

    # Precompute template stats
    t_norm = template / (np.linalg.norm(template) + 1e-10)

    for r in range(nrows):
        for c in range(ncols):
            patch = gz_pad[r:r+t_size, c:c+t_size]
            patch_mean = patch.mean()
            patch_centered = patch - patch_mean
            patch_norm = np.linalg.norm(patch_centered)
            if patch_norm < 1e-10:
                ncc[r, c] = 0.0
            else:
                ncc[r, c] = np.dot(patch_centered.ravel(), t_norm.ravel()) / patch_norm

    return ncc


def _fast_normalized_xcorr(grid: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Proper Normalized Cross-Correlation (NCC) ∈ [-1, +1].

    NCC(r,c) = Σ[(gz_patch - μ_patch) · (t - μ_t)] / (σ_patch · σ_t · N)

    Uses scipy.ndimage.uniform_filter for fast local mean/variance estimation.
    Output is strictly bounded [-1, 1] — cosine similarity between centered
    local patch and centered template.
    """
    if template.size == 0:
        return np.zeros_like(grid)
    t_size = template.shape[0]
    N = t_size * t_size

    # Center and normalize template
    t_centered = template - template.mean()
    t_norm_val = np.linalg.norm(t_centered)
    if t_norm_val < 1e-10:
        return np.zeros_like(grid)
    t_normalized = t_centered / t_norm_val

    # Cross-correlation with centered, normalized template
    corr = signal.correlate2d(grid, t_normalized, mode="same", boundary="symm")

    # Local statistics via uniform filter (fast box filter)
    local_mean  = ndimage.uniform_filter(grid.astype(float), size=t_size)
    local_sqmean = ndimage.uniform_filter(grid.astype(float)**2, size=t_size)
    local_var   = np.maximum(local_sqmean - local_mean**2, 0.0)

    # Local signal norm: sqrt(Var) * sqrt(N) = local std × sqrt(N)
    local_norm = np.sqrt(local_var) * np.sqrt(N)

    # Floor: at least 1% of grid std to avoid division by near-zero flat regions
    grid_std_floor = max(float(np.std(grid)) * 0.01, 1e-4)
    local_norm = np.where(local_norm < grid_std_floor, grid_std_floor, local_norm)

    ncc = corr / local_norm
    return np.clip(ncc, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────────────────────────────────────

class MatchedDipoleDetector(AnomalyDetectorBase):
    """
    Matched Filter detector using a bank of magnetic dipole templates.

    For each template in the bank, computes normalized cross-correlation
    with the processed grid. Takes the maximum response across templates.

    A high correlation peak means the local field pattern closely matches
    the expected signature of a buried ferrous object — regardless of
    whether the curvature (LoG) is high or not.

    This detector works equally well on:
      - griddata_cubic output (moderate smoothness)
      - RBF output (C∞ smooth) — because NCC is not curvature-dependent

    Parameters:
      ncc_threshold: minimum NCC score to flag a blob [0, 1]
        0.35 → moderate match (more sensitive)
        0.50 → strong match (fewer false positives)
      min_depth_cells: template minimum depth [grid cells]
      max_depth_cells: template maximum depth
      n_depths: number of template scales
      n_orientations: number of horizontal dipole orientations
      template_size: pixel size of each template (odd integer)
      fast_ncc: use fast (scipy.correlate2d) vs exact sliding window NCC
    """

    def __init__(self,
                 ncc_threshold: float = 0.40,
                 min_depth_cells: float = 2.0,
                 max_depth_cells: float = 8.0,
                 n_depths: int = 4,
                 n_orientations: int = 4,
                 template_size: int = 15,
                 fast_ncc: bool = True):
        self.ncc_threshold    = ncc_threshold
        self.min_depth_cells  = min_depth_cells
        self.max_depth_cells  = max_depth_cells
        self.n_depths         = n_depths
        self.n_orientations   = n_orientations
        self.template_size    = template_size + (1 - template_size % 2)  # force odd
        self.fast_ncc         = fast_ncc
        self._template_bank   = None  # lazy-initialized

    @property
    def name(self) -> str:
        return "matched_dipole"

    @property
    def compatibility(self) -> StageCompatibility:
        from ..abstractions import StageCompatibility
        return StageCompatibility(
            name=self.name,
            notes=(
                "Works with ALL interpolators (cubic, RBF) because NCC is "
                "not curvature-dependent. Prefers NoBaseline or LineMedian."
            )
        )

    def _get_bank(self) -> list:
        if self._template_bank is None:
            self._template_bank = _build_template_bank(
                min_depth=self.min_depth_cells,
                max_depth=self.max_depth_cells,
                n_depths=self.n_depths,
                n_orientations=self.n_orientations,
                template_size=self.template_size,
            )
        return self._template_bank

    def detect(self, grid: BaselinedGrid, config: dict) -> DetectionResult:
        ad = config.get("anomaly_detection", {})
        min_extent = ad.get("min_spatial_extent", 5)
        rules      = config.get("classification", {})

        gz   = grid.grid_z
        mask = grid.grid_mask
        nf   = grid.noise_floor

        logger.info(f"[MatchedDipole] detecting in {grid.scan_id}")

        bank = self._get_bank()
        ncc_fn = _fast_normalized_xcorr if self.fast_ncc else _normalized_xcorr

        # Compute NCC for each template, keep maximum response
        ncc_max = np.zeros_like(gz)
        best_template = np.full(gz.shape, "", dtype=object)

        for tmpl_name, tmpl in bank:
            ncc = ncc_fn(gz, tmpl) * mask
            improved = ncc > ncc_max
            ncc_max  = np.where(improved, ncc, ncc_max)
            best_template = np.where(improved, tmpl_name, best_template)

        # Also keep a LoG response layer for hybrid SNR scoring
        from scipy import ndimage as nd
        sigmas = ad.get("multi_scale_sigmas", [1.0, 2.0, 4.0])
        responses = [nd.gaussian_laplace(gz, s) * s**2 for s in sigmas]
        log_resp  = np.max(np.abs(np.stack(responses)), axis=0) * mask
        log_snr   = np.where(nf > 0, log_resp / nf, 0)

        # Hybrid detection map: NCC threshold OR LoG backup
        binary = (
            (ncc_max >= self.ncc_threshold) |
            (log_snr >= ad.get("snr_min", 2.6))
        ).astype(int) * mask

        labeled, n = ndimage.label(binary)
        logger.debug(f"  {n} candidate blobs (NCC≥{self.ncc_threshold} | LoG backup)")

        anomalies = []
        for idx in range(1, n+1):
            blob_mask = labeled == idx
            extent = int(blob_mask.sum())
            if extent < min_extent:
                continue

            region = gz[blob_mask]
            rows, cols = np.where(blob_mask)
            bbox = (int(rows.min()), int(cols.min()),
                    int(rows.max()), int(cols.max()))

            r0 = max(0, bbox[0]-8); c0 = max(0, bbox[1]-8)
            r1 = min(gz.shape[0], bbox[2]+9); c1 = min(gz.shape[1], bbox[3]+9)
            sub = gz[r0:r1, c0:c1]

            snr = _robust_snr(region, nf)
            sm  = _smoothness(sub)
            dip = _dipole_score(sub)
            pol = _polarity(sub)
            coh = _coherence(blob_mask, gz)
            fs  = _final_score(dip, pol, coh, sm)
            unc = _uncertainty(snr, sm, extent, min_extent)

            # Boost confidence if high NCC match
            peak_ncc = float(ncc_max[blob_mask].max())
            ncc_boost = float(np.clip(peak_ncc * 0.20, 0, 0.20))

            label, conf = _classify(snr, sm, dip, pol, coh, fs, unc, rules)
            conf = round(float(np.clip(conf + ncc_boost, 0, 1)), 3)

            # Best template match for this blob
            template_match = str(best_template[blob_mask][
                np.argmax(ncc_max[blob_mask])
            ])

            cy_b = float(np.mean(rows)); cx_b = float(np.mean(cols))
            if label == "FERROUS_METAL":
                mcy, mcx = _dipole_midpoint(sub, r0, c0)
            else:
                mcy, mcx = cy_b, cx_b

            anomalies.append(RawAnomaly(
                anomaly_id=f"{grid.scan_id}_MF{idx:03d}",
                cx=round(cx_b,2), cy=round(cy_b,2),
                marker_cx=round(mcx,2), marker_cy=round(mcy,2),
                extent_cells=extent,
                peak_amplitude=round(float(np.abs(region).max()),2),
                snr_robust=round(snr,3),
                smoothness_score=round(sm,3),
                dipole_score=round(dip,3),
                polarity_ratio=round(pol,3),
                spatial_coherence=round(coh,3),
                final_score=round(fs,3),
                uncertainty=round(unc,3),
                raw_label=label,
                confidence=conf,
                bbox=bbox,
                detector_name=f"matched_dipole(ncc={peak_ncc:.2f},t={template_match})",
            ))

        logger.info(f"  {len(anomalies)} valid anomalies (MatchedDipole)")

        return DetectionResult(
            scan_id=grid.scan_id,
            anomalies=anomalies,
            scan_quality_score=_quality_score(mask, grid.dynamic_range),
            noise_floor=nf,
            detector_name=self.name,
            warnings=list(grid.warnings),
        )
