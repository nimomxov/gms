"""
GMS — Multi-Scale Adaptive Baseline  v2.2

The core problem with rolling median at a single window size:
  - Small window → absorbs compact anomalies
  - Large window → misses slow drift, absorbs broad cavities
  - No single window size works for all anomaly types.

Solution: multi-scale decomposition.

Algorithm:
  1. Decompose the signal into multiple scales using Gaussian smoothing
  2. At each scale, estimate what is "background" vs "anomaly"
  3. The background at each scale = morphological opening
     (erosion followed by dilation — preserves positive blobs, removes negatives)
  4. Combine scale estimates into an adaptive background model
  5. Subtract background → residual = anomalies of all scales

This preserves:
  - Sharp positive dipoles (scale 1-3 cells)
  - Broad negative cavities (scale 10-30 cells)
  - While removing:
    - Sensor drift along traverse lines
    - Broad positive basalt gradients
    - DC baseline offsets

Physical motivation:
  The "background" field is the Earth's slowly varying magnetic field
  plus sensor drift. Anomalies are deviations from this background.
  Morphological opening estimates the minimum envelope of the signal
  at each scale, which approximates the background for positive targets.
  For negative targets (cavities), we use morphological closing.
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation

from ..abstractions import BaselineRemoverBase, GriddedScan, BaselinedGrid

logger = logging.getLogger("gms.multiscale_baseline")


def _morphological_opening(grid: np.ndarray, radius: int) -> np.ndarray:
    """
    2D morphological opening with circular structuring element.
    Opening = erosion then dilation.
    Removes positive blobs smaller than 'radius' cells.
    Preserves broad positive backgrounds.
    """
    struct = _disk_struct(radius)
    eroded  = ndimage.grey_erosion(grid,  footprint=struct)
    opened  = ndimage.grey_dilation(eroded, footprint=struct)
    return opened


def _morphological_closing(grid: np.ndarray, radius: int) -> np.ndarray:
    """
    2D morphological closing with circular structuring element.
    Closing = dilation then erosion.
    Fills negative blobs (cavities) smaller than 'radius' cells.
    Preserves broad negative backgrounds.
    """
    struct = _disk_struct(radius)
    dilated = ndimage.grey_dilation(grid,  footprint=struct)
    closed  = ndimage.grey_erosion(dilated, footprint=struct)
    return closed


def _disk_struct(radius: int) -> np.ndarray:
    """Circular binary structuring element of given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (x**2 + y**2) <= radius**2


def _robust_background(grid: np.ndarray, mask: np.ndarray,
                        radius: int) -> np.ndarray:
    """
    Estimate background at a given scale using morphological opening + closing.
    The background is the average of the opening (positive bias) and
    closing (negative bias) — this is neutral to both target polarities.
    """
    gz_masked = grid.copy()
    gz_masked[~mask] = 0.0

    bg_pos = _morphological_opening(gz_masked, radius)   # positive background
    bg_neg = _morphological_closing(gz_masked, radius)   # negative background
    background = 0.5 * (bg_pos + bg_neg)
    background[~mask] = 0.0
    return background


class MultiScaleAdaptiveBaseline(BaselineRemoverBase):
    """
    Multi-scale morphological baseline removal.

    Computes background at multiple spatial scales and combines them
    into an adaptive background estimate that preserves anomalies
    at ALL scales (compact dipoles AND broad cavities).

    Parameters:
      scales_cells: list of morphological radii [grid cells]
        Small radii (2-5):   remove compact artifacts, preserve broad features
        Large radii (10-25): remove broad background, preserve large anomalies
        Include a range to handle all anomaly sizes.

      scale_weights: weight for each scale's background estimate
        Default: uniform weights
        Set higher weight on larger scales to remove more broad background.

      pre_line_dc: apply per-line DC offset before morphological baseline

    Compared to rolling median:
      - Preserves broad cavities (closing preserves negative background)
      - Preserves sharp dipoles (opening at small scale doesn't absorb them)
      - No window size tuning needed
      - Slightly slower (~0.5s for 100×80 grid at 3 scales)
    """

    def __init__(self,
                 scales_cells: list = None,
                 scale_weights: list = None,
                 pre_line_dc: bool = True):
        self.scales_cells  = scales_cells  or [3, 8, 18]
        self.scale_weights = scale_weights or None   # auto-uniform
        self.pre_line_dc   = pre_line_dc

    @property
    def name(self) -> str:
        return "multiscale"

    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        gz   = grid.grid_z.copy()
        mask = grid.grid_mask
        warnings = list(grid.warnings)

        # Step 1: Per-line DC offset (remove traverse-direction bias)
        if self.pre_line_dc:
            for row_idx in range(gz.shape[0]):
                valid = mask[row_idx]
                if valid.sum() > 3:
                    gz[row_idx, valid] -= np.median(gz[row_idx, valid])
            logger.debug("  Multiscale: per-line DC applied")

        # Step 2: Compute background at each scale
        scales  = self.scales_cells
        weights = (self.scale_weights if self.scale_weights
                   else [1.0/len(scales)] * len(scales))
        weights = np.array(weights)
        weights /= weights.sum()

        background = np.zeros_like(gz)
        for w, radius in zip(weights, scales):
            bg = _robust_background(gz, mask, radius)
            background += w * bg

        # Step 3: Residual = signal - background
        residual = gz - background
        residual[~mask] = 0.0

        # Recompute noise floor and dynamic range
        valid_vals = residual[mask]
        nf = float(median_abs_deviation(valid_vals)) if valid_vals.size > 3 else 0.0
        dr = float(np.ptp(valid_vals)) if valid_vals.size > 0 else 0.0

        logger.info(
            f"  Multiscale baseline: scales={scales}  "
            f"noise_floor={nf:.3f}  dynamic_range={dr:.2f}"
        )

        return BaselinedGrid(
            scan_id=grid.scan_id,
            grid_z=residual,
            grid_x=grid.grid_x,
            grid_y=grid.grid_y,
            grid_mask=mask,
            noise_floor=nf,
            dynamic_range=dr,
            baseline_name=self.name,
            interp_name=grid.interp_name,
            warnings=warnings,
            meta=dict(grid.meta),
        )


class WaveletBaseline(BaselineRemoverBase):
    """
    Wavelet-inspired background estimation using Gaussian scale-space.

    At each octave, subtract the low-frequency component estimated
    by a Gaussian smooth. The residual is the anomaly signal.

    Simpler and faster than morphological approach.
    Less physically motivated but good for clean data.

    Sigma = 25% of grid min-dimension → removes broad trends.
    """

    def __init__(self, sigma_fraction: float = 0.20):
        self.sigma_fraction = sigma_fraction

    @property
    def name(self) -> str:
        return "wavelet_bg"

    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        from scipy.stats import median_abs_deviation as mad_fn
        gz   = grid.grid_z.copy()
        mask = grid.grid_mask

        sigma = max(3.0, self.sigma_fraction * min(gz.shape))
        background = ndimage.gaussian_filter(gz, sigma=sigma)
        residual = gz - background
        residual[~mask] = 0.0

        valid = residual[mask]
        nf = float(mad_fn(valid)) if valid.size > 3 else 0.0
        dr = float(np.ptp(valid)) if valid.size > 0 else 0.0

        logger.debug(f"  Wavelet baseline: sigma={sigma:.1f}  nf={nf:.3f}  dr={dr:.2f}")

        return BaselinedGrid(
            scan_id=grid.scan_id,
            grid_z=residual,
            grid_x=grid.grid_x, grid_y=grid.grid_y,
            grid_mask=mask,
            noise_floor=nf, dynamic_range=dr,
            baseline_name=self.name, interp_name=grid.interp_name,
            warnings=list(grid.warnings), meta=dict(grid.meta),
        )
