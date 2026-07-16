"""
GMS Demo Data Generator
Produces synthetic geophysical scan CSVs with realistic anomalies.

Generates:
  - scan_A.csv : contains ferrous metal anomaly + background noise
  - scan_B.csv : same anomaly shifted slightly (cross-scan confirmation)
  - scan_C.csv : noise-only scan (no real target)

Usage:
  python demo_data/generate_demo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent


def _background(n: int, level: float = 512.0, noise_std: float = 18.0) -> np.ndarray:
    """Gaussian baseline with gentle drift."""
    drift = np.linspace(0, 12, n)
    return level + drift + RNG.normal(0, noise_std, n)


def _ferrous_metal_dipole(x: np.ndarray, y: np.ndarray,
                           cx: float, cy: float,
                           amplitude: float = 120.0,
                           width: float = 0.8) -> np.ndarray:
    """
    Simulate ferrous metal dipole signature:
    positive lobe above center + negative lobe below.
    """
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    # Positive lobe
    pos = amplitude * np.exp(-r2 / (2 * width ** 2))
    # Negative lobe offset in y
    r2_neg = (x - cx) ** 2 + (y - (cy + width * 1.5)) ** 2
    neg = -0.6 * amplitude * np.exp(-r2_neg / (2 * (width * 1.2) ** 2))
    return pos + neg


def _cavity_signature(x: np.ndarray, y: np.ndarray,
                       cx: float, cy: float,
                       amplitude: float = -60.0,
                       width: float = 1.2) -> np.ndarray:
    """Cavity: broad, smooth negative anomaly (reduced signal)."""
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amplitude * np.exp(-r2 / (2 * width ** 2))


def generate_scan(x_range=(0, 10), y_range=(0, 8),
                  n_lines: int = 20, pts_per_line: int = 60,
                  anomaly_fn=None, noise_std=18.0,
                  filename: str = "scan.csv"):
    """Generate a grid scan CSV with optional injected anomaly."""
    rows = []
    x_vals = np.linspace(x_range[0], x_range[1], pts_per_line)

    for line_idx in range(n_lines):
        y = y_range[0] + (y_range[1] - y_range[0]) * line_idx / (n_lines - 1)
        y_arr = np.full_like(x_vals, y)
        bg = _background(pts_per_line, noise_std=noise_std)

        signal = bg.copy()
        if anomaly_fn is not None:
            signal += anomaly_fn(x_vals, y_arr)

        # Clip to ADC range [0, 1024]
        signal = np.clip(signal, 0, 1024)

        for x, v in zip(x_vals, signal):
            rows.append({"x": round(x, 3), "y": round(y, 3),
                         "value": round(v, 2)})

    df = pd.DataFrame(rows)
    out_path = OUT / filename
    df.to_csv(out_path, index=False)
    print(f"  Generated: {out_path}  ({len(df)} samples)")
    return str(out_path)


def main():
    print("GMS Demo Data Generator")
    print("=" * 40)

    # Scan A: ferrous metal at (5, 4) + small cavity hint at (2, 6)
    def anomaly_A(x, y):
        return (
            _ferrous_metal_dipole(x, y, cx=5.0, cy=4.0, amplitude=320) +
            _cavity_signature(x, y, cx=2.0, cy=6.0, amplitude=-110)
        )

    generate_scan(
        anomaly_fn=anomaly_A,
        noise_std=20.0,
        filename="scan_A.csv"
    )

    # Scan B: same anomalies, slightly shifted position (simulates re-scan)
    def anomaly_B(x, y):
        return (
            _ferrous_metal_dipole(x, y, cx=5.1, cy=3.9, amplitude=300) +
            _cavity_signature(x, y, cx=2.1, cy=6.1, amplitude=-115)
        )

    generate_scan(
        anomaly_fn=anomaly_B,
        noise_std=22.0,
        filename="scan_B.csv"
    )

    # Scan C: noise only — no real target
    generate_scan(
        anomaly_fn=None,
        noise_std=18.0,
        filename="scan_C_noise_only.csv"
    )

    print("\nDemo scans ready. Run the full pipeline with:")
    print("  python main.py --scans demo_data/scan_A.csv demo_data/scan_B.csv demo_data/scan_C_noise_only.csv")


if __name__ == "__main__":
    main()
