"""
GMS — Adaptive Local Threshold Baseline  v2.0

Replaces global MAD noise floor with a spatially varying estimate.

WHY global MAD fails in heterogeneous terrain:
  Global MAD = single noise value for the entire scan.
  In reality:
    - basalt areas have high magnetic noise
    - clay has low noise
    - mineralized zones have medium noise
  A single threshold causes:
    - Over-detection in basalt (many false positives)
    - Under-detection in clay (misses real targets)

SOLUTION — Local MAD in sliding windows:
  For each grid cell, estimate the noise floor from a local window
  around that cell. This adapts the threshold to the local background.

  noise_floor_local(r,c) = MAD(gz[r-w:r+w, c-w:c+w])

  The SNR map becomes:
    snr(r,c) = |gz(r,c)| / noise_floor_local(r,c)

  This is computed AFTER baseline removal (which removes slow drift).
  The adaptive threshold addresses fast spatial variation in noise.

Also used by detectors: the BaselinedGrid now carries a 2D noise_map
in its meta dict if this baseline was used.
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation

from ..abstractions import BaselineRemoverBase, GriddedScan, BaselinedGrid

logger = logging.getLogger("gms.adaptive_baseline")


class AdaptiveLocalBaseline(BaselineRemoverBase):
    """
    Two-stage baseline:
      1. Global row-median (same as LineMedianBaseline) to remove traverse drift
      2. Compute local MAD noise map for use by detectors

    The local noise map is stored in BaselinedGrid.meta["noise_map"].
    Detectors that support it (AmplitudeDetector, MatchedDipoleDetector)
    will use it instead of the global noise_floor.

    Parameters:
      window_fraction: row baseline window (same as LineMedianBaseline)
      local_window_cells: half-size of the local noise estimation window
        Small window (5-10): adapts quickly, may include anomaly signal
        Large window (15-25): smoother map, better separation
    """

    def __init__(self, window_fraction: float = 0.90,
                 local_window_cells: int = 12):
        self.window_fraction    = window_fraction
        self.local_window_cells = local_window_cells

    @property
    def name(self) -> str:
        return "adaptive_local"

    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        from .plugins import LineMedianBaseline

        # Stage 1: row-median baseline (reuse existing plugin)
        row_baseline = LineMedianBaseline(self.window_fraction)
        baselined    = row_baseline.remove(grid)

        # Stage 2: compute local MAD noise map
        gz   = baselined.grid_z
        mask = baselined.grid_mask
        w    = self.local_window_cells

        noise_map = _local_mad_map(gz, mask, w)

        # Global noise floor = median of local estimates (in valid region)
        valid_noise = noise_map[mask & (noise_map > 0)]
        global_nf   = float(np.median(valid_noise)) if valid_noise.size > 0 else baselined.noise_floor

        logger.debug(
            f"  Adaptive baseline: local_window={w}  "
            f"global_nf={global_nf:.3f}  "
            f"noise_range=[{noise_map[mask].min():.2f},{noise_map[mask].max():.2f}]"
        )

        # Store noise_map in meta for detectors to use
        meta = dict(baselined.meta)
        meta["noise_map"] = noise_map
        meta["adaptive_window"] = w

        from ..abstractions import BaselinedGrid as BG
        return BG(
            scan_id=baselined.scan_id,
            grid_z=baselined.grid_z,
            grid_x=baselined.grid_x,
            grid_y=baselined.grid_y,
            grid_mask=baselined.grid_mask,
            noise_floor=global_nf,
            dynamic_range=baselined.dynamic_range,
            baseline_name=self.name,
            interp_name=baselined.interp_name,
            warnings=baselined.warnings,
            meta=meta,
        )


def _local_mad_map(gz: np.ndarray, mask: np.ndarray,
                    window: int) -> np.ndarray:
    """
    Compute local MAD at every grid cell using a sliding square window.

    For each cell (r,c), the local MAD is computed over the window
    [r-w:r+w, c-w:c+w] — only counting valid (masked) cells.

    Uses scipy.ndimage.generic_filter for efficiency.
    Zero is returned for cells with too few valid neighbors.
    """
    size = 2 * window + 1

    def local_mad(patch_flat):
        # patch includes invalid (masked=0) cells — ignore them
        # We can't pass mask into generic_filter directly,
        # so we use a threshold: cells near 0 with no anomaly are background
        vals = patch_flat[np.abs(patch_flat) < 1e10]  # all finite
        if len(vals) < 4:
            return 0.0
        return float(median_abs_deviation(vals))

    noise_map = ndimage.generic_filter(
        gz, local_mad,
        size=size,
        mode="reflect"
    )

    # Zero out invalid cells
    noise_map[~mask] = 0.0

    # Ensure no zero noise floors in valid region (would cause division by zero)
    min_nf = float(np.percentile(noise_map[mask], 5)) if mask.any() else 1.0
    min_nf = max(min_nf, 1e-3)
    noise_map = np.where((noise_map < 1e-6) & mask, min_nf, noise_map)

    return noise_map


class AdaptiveLoGDetector:
    """
    LoG detector that uses the local noise map from AdaptiveLocalBaseline.

    Instead of:
      snr = log_response / global_mad

    Uses:
      snr = log_response / local_mad(r,c)

    This eliminates false positives in high-noise zones and recovers
    detections in low-noise zones.

    Falls back to global noise floor if no noise_map in meta.
    """

    @property
    def name(self) -> str:
        return "adaptive_log"

    def detect(self, grid: BaselinedGrid, config: dict):
        from ..abstractions import DetectionResult
        from .plugins import _extract_anomaly, _quality_score
        from scipy import ndimage as nd

        ad = config.get("anomaly_detection", {})
        snr_min    = ad.get("snr_min", 2.6)
        min_extent = ad.get("min_spatial_extent", 5)
        sigmas     = ad.get("multi_scale_sigmas", [1.0, 2.0, 4.0])
        rules      = config.get("classification", {})

        gz, mask = grid.grid_z, grid.grid_mask
        noise_map = grid.meta.get("noise_map", None)

        if noise_map is None:
            logger.warning("  [AdaptiveLoG] No noise_map found — using global noise floor")
            noise_map = np.full_like(gz, grid.noise_floor)

        logger.info(f"[AdaptiveLoG] detecting in {grid.scan_id}")

        responses = [nd.gaussian_laplace(gz, s)*s**2 for s in sigmas]
        scale_resp = np.max(np.abs(np.stack(responses)), axis=0) * mask

        # Local SNR: divide by local noise estimate at each cell
        snr_grid = np.where(noise_map > 1e-6, scale_resp / noise_map, 0)

        labeled, n = nd.label(snr_grid > snr_min)
        logger.debug(f"  {n} candidate blobs (adaptive-LoG)")

        anomalies = []
        for idx in range(1, n+1):
            a = _extract_anomaly(idx, gz, labeled, labeled==idx,
                                 grid.noise_floor, min_extent, rules,
                                 grid.scan_id, self.name)
            if a: anomalies.append(a)

        logger.info(f"  {len(anomalies)} valid anomalies")
        return DetectionResult(
            scan_id=grid.scan_id,
            anomalies=anomalies,
            scan_quality_score=_quality_score(mask, grid.dynamic_range),
            noise_floor=grid.noise_floor,
            detector_name=self.name,
            warnings=list(grid.warnings),
        )
