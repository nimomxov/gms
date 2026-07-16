"""
GMS Core — Signal Processing Module v1.4 (PATCHED)
FIX 4: grid axis build now guards zero-span / collinear input.
FIX 6: RBF smoothing comment standardized to the real formula (0.5*noise_floor).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, interpolate
from scipy.interpolate import RBFInterpolator
from scipy.stats import median_abs_deviation
from scipy.ndimage import uniform_filter1d

from .ingestion import ScanDataset

logger = logging.getLogger("gms.signal")


@dataclass
class ProcessedGrid:
    scan_id: str
    grid_z: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_mask: np.ndarray
    baseline: float
    noise_floor: float
    dynamic_range: float
    drift_method: str
    interp_method: str
    warnings: list


def _mad_zscore(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = median_abs_deviation(values, nan_policy="omit")
    if mad < 1e-10:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def _linebased_robust_baseline(x, y, v, window_fraction: float = 0.75):
    v_clean = v.copy()
    for y_val in np.unique(y):
        mask = y == y_val
        v_clean[mask] = v[mask] - np.median(v[mask])
    return v_clean


def _linebased_baseline_2d(grid_z, grid_mask, window_fraction: float = 0.75):
    result = grid_z.copy()
    nrows, ncols = grid_z.shape
    for row_idx in range(nrows):
        row = grid_z[row_idx].copy()
        valid = grid_mask[row_idx]
        if valid.sum() < 4:
            result[row_idx] = row
            continue
        valid_cols = np.where(valid)[0]
        v_valid = row[valid_cols]
        win = max(3, int(len(v_valid) * window_fraction) | 1)
        baseline_valid = np.array([
            np.median(v_valid[max(0, i - win//2): i + win//2 + 1])
            for i in range(len(v_valid))
        ])
        edge = win // 4
        if edge > 0 and len(baseline_valid) > 2 * edge:
            row_med = np.median(v_valid)
            for k in range(edge):
                alpha = k / edge
                baseline_valid[k] = alpha * baseline_valid[k] + (1-alpha) * row_med
                baseline_valid[-(k+1)] = alpha * baseline_valid[-(k+1)] + (1-alpha) * row_med
        corrected = np.zeros(ncols)
        corrected[valid_cols] = v_valid - baseline_valid
        result[row_idx] = corrected
    return result


def _highpass_2d(grid_z, sigma_fraction: float = 0.15):
    sigma = max(2.0, sigma_fraction * max(grid_z.shape))
    low_freq = ndimage.gaussian_filter(grid_z, sigma=sigma)
    return grid_z - low_freq


def _polynomial_detrend(x, y, v, degree=2):
    coeffs = np.polyfit(x, v, degree)
    trend = np.polyval(coeffs, x)
    v2 = v - trend
    coeffs2 = np.polyfit(y, v2, 1)
    return v2 - np.polyval(coeffs2, y)


def _rbf_interpolate(xv, yv, zv, xi, yi, noise_floor, kernel="thin_plate_spline"):
    """
    RBF interpolation onto a regular grid.
    FIX 6: smoothing = 0.5 * noise_floor. Near-exact interpolation that damps
    only spikes on the order of the noise floor. This is NOT noise variance
    and NOT n * variance — the code and this comment now agree.
    """
    smoothing = max(0.1, float(noise_floor) * 0.5)
    grid_x, grid_y = np.meshgrid(xi, yi)
    query_pts = np.c_[grid_x.ravel(), grid_y.ravel()]
    train_pts = np.c_[xv, yv]
    try:
        rbf = RBFInterpolator(train_pts, zv, kernel=kernel, smoothing=smoothing, degree=1)
        grid_z = rbf(query_pts).reshape(grid_x.shape)
        return grid_z, f"RBF({kernel}, smoothing={smoothing:.1f})"
    except Exception as e:
        logger.warning(f"  RBF({kernel}) failed: {e} — falling back to griddata cubic")
        try:
            grid_z = interpolate.griddata(train_pts, zv, (grid_x, grid_y), method="cubic")
            return np.nan_to_num(grid_z, nan=0.0), "griddata_cubic_fallback"
        except Exception as e2:
            logger.warning(f"  griddata cubic failed: {e2} — falling back to linear")
            grid_z = interpolate.griddata(train_pts, zv, (grid_x, grid_y), method="linear")
            return np.nan_to_num(grid_z, nan=0.0), "griddata_linear_fallback"


class SignalProcessor:
    def __init__(self, config: dict):
        sp = config.get("signal_processing", {})
        self.drift_cfg = sp.get("drift_removal", {})
        self.noise_cfg = sp.get("noise_filter", {})
        self.grid_cfg = sp.get("grid_interpolation", {})

    def process(self, dataset: ScanDataset) -> ProcessedGrid:
        logger.info(f"Processing scan: {dataset.scan_id}")
        warnings = list(dataset.warnings)

        x = dataset.x.copy(); y = dataset.y.copy(); v = dataset.values.copy()

        # 1. Global DC removal
        global_baseline = float(np.median(v))
        v = v - global_baseline

        # 2. Drift removal (pre-grid)
        method = self.drift_cfg.get("method", "median_line")
        drift_method_used = method
        if not self.drift_cfg.get("enabled", True):
            v_clean = v.copy(); drift_method_used = "none"
        elif method == "median_line":
            v_clean = _linebased_robust_baseline(x, y, v)
        elif method == "polynomial":
            degree = self.drift_cfg.get("poly_degree", 2)
            v_clean = _polynomial_detrend(x, y, v, degree)
            warnings.append("Polynomial detrending used — may absorb dipole lobes. "
                            "Recommend: method: median_line")
        elif method == "median":
            v_clean = v - np.median(v)
        else:
            v_clean = v.copy()
            warnings.append(f"Unknown drift method '{method}' — skipped")

        # 3. MAD outlier rejection
        threshold = self.noise_cfg.get("mad_threshold", 3.5)
        zscores = _mad_zscore(v_clean)
        valid = np.abs(zscores) < threshold
        n_rej = int((~valid).sum())
        if n_rej > 0:
            frac = n_rej / len(v_clean)
            if frac > 0.10:
                warnings.append(f"High outlier rate: {frac:.1%} rejected")
        xv, yv, zv = x[valid], y[valid], v_clean[valid]

        # 4. Grid setup — FIX 4: guard degenerate geometry
        resolution = self.grid_cfg.get("resolution", 0.1)
        x_span = float(xv.max() - xv.min())
        y_span = float(yv.max() - yv.min())
        if x_span < 1e-9 or y_span < 1e-9:
            raise ValueError(
                "Degenerate scan geometry: zero spatial span on one axis. "
                "This device likely has no valid x/y — route to line mode."
            )
        n_x = max(2, min(500, int(x_span / resolution) + 1))
        n_y = max(2, min(500, int(y_span / resolution) + 1))
        xi = np.linspace(xv.min(), xv.max(), n_x)
        yi = np.linspace(yv.min(), yv.max(), n_y)

        # 5. Grid interpolation
        noise_floor_scatter = float(median_abs_deviation(zv)) if len(zv) > 3 else 10.0
        interp_mode = self.grid_cfg.get("mode", "cubic")
        if interp_mode == "rbf":
            rbf_kernel = self.grid_cfg.get("rbf_kernel", "thin_plate_spline")
            grid_z, interp_method_used = _rbf_interpolate(
                xv, yv, zv, xi, yi, noise_floor=noise_floor_scatter, kernel=rbf_kernel)
        else:
            grid_X, grid_Y = np.meshgrid(xi, yi)
            try:
                grid_z = interpolate.griddata(np.c_[xv, yv], zv, (grid_X, grid_Y), method="cubic")
            except Exception as e:
                logger.warning(f"  griddata cubic failed: {e} — fallback linear")
                grid_z = interpolate.griddata(np.c_[xv, yv], zv, (grid_X, grid_Y), method="linear")
            interp_method_used = "griddata_cubic"

        # Build validity mask
        from scipy.spatial import Delaunay
        try:
            hull = Delaunay(np.c_[xv, yv])
            grid_X2, grid_Y2 = np.meshgrid(xi, yi)
            inside = hull.find_simplex(np.c_[grid_X2.ravel(), grid_Y2.ravel()]) >= 0
            grid_mask = inside.reshape(grid_X2.shape)
        except Exception:
            grid_mask = ~np.isnan(grid_z)

        grid_z = np.nan_to_num(grid_z, nan=0.0)
        grid_z[~grid_mask] = 0.0

        # 6. Post-grid drift baseline
        if method == "median_line":
            win_frac = self.drift_cfg.get("window_fraction", 0.90)
            grid_z = _linebased_baseline_2d(grid_z, grid_mask, window_fraction=win_frac)
            grid_z[~grid_mask] = 0.0
        elif method == "highpass":
            hp_sigma_frac = self.drift_cfg.get("highpass_sigma_fraction", 0.15)
            grid_z = _highpass_2d(grid_z, sigma_fraction=hp_sigma_frac)
            grid_z[~grid_mask] = 0.0
            drift_method_used = "highpass_2d"

        # 7. Light smoothing
        kernel = self.noise_cfg.get("kernel_size", 3)
        grid_z_clean = ndimage.gaussian_filter(grid_z, sigma=kernel / 4.0)
        grid_z_clean[~grid_mask] = 0.0

        # 8. Noise floor on grid
        valid_vals = grid_z_clean[grid_mask]
        noise_floor = float(median_abs_deviation(valid_vals)) if valid_vals.size > 0 else noise_floor_scatter
        dynamic_range = float(np.ptp(valid_vals)) if valid_vals.size > 0 else 0.0

        return ProcessedGrid(
            scan_id=dataset.scan_id, grid_z=grid_z_clean, grid_x=xi, grid_y=yi,
            grid_mask=grid_mask, baseline=round(global_baseline, 3),
            noise_floor=noise_floor, dynamic_range=dynamic_range,
            drift_method=drift_method_used, interp_method=interp_method_used,
            warnings=warnings)