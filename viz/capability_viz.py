"""
GMS — Capability-Gated Visualization Engine  v2.3

Selects and renders the appropriate visualization based on DeviceCapabilities.

PRIMARY VIEW:  Scientific 2D heatmap (requires position)
SECONDARY VIEW: Interactive 3D explorer (requires position)
FALLBACK VIEW: Line signal trace (no position required)

Features enabled only when telemetry supports them:
  - Anomaly contours        → requires position
  - Dig markers             → requires anomaly detection result
  - Confidence rings        → requires SNR
  - Uncertainty radius      → requires SNR
  - Crosshair + labels      → always
  - Scan grid overlay       → requires position
  - Target IDs              → requires anomaly detection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib import cm

from core.schema.capabilities import DeviceCapabilities, TelemetryGrade
from core.pipeline_composer import ComposedPipeline
from core.abstractions import DetectionResult, BaselinedGrid, RawAnomaly

logger = logging.getLogger("gms.viz.engine")

# ─────────────────────────────────────────────────────────────────────────────
# Color scheme (matching existing system)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "FERROUS_METAL":  "#FF2222",
    "CAVITY":         "#2266FF",
    "ROCK_DEBRIS":    "#CC8800",
    "SOIL_VARIATION": "#6B8E23",
    "NOISE":          "#AAAAAA",
    "UNKNOWN":        "#CCCCCC",
}

DIG_SQUARE_COLORS = {
    "FERROUS_METAL": "#FFFFFF",
    "CAVITY":        "#44AAFF",
    "ROCK_DEBRIS":   "#FFCC44",
}

GRADE_BADGE = {
    TelemetryGrade.BASIC:        ("#555555", "BASIC"),
    TelemetryGrade.STANDARD:     ("#1a6bbf", "STANDARD"),
    TelemetryGrade.ADVANCED:     ("#1f8c4e", "ADVANCED"),
    TelemetryGrade.PROFESSIONAL: ("#b8860b", "PROFESSIONAL"),
}


class CapabilityGatedVizEngine:
    """
    Routes visualization to the appropriate renderer based on capabilities.

    Usage:
        engine = CapabilityGatedVizEngine(config, output_dir="reports")
        paths = engine.render(grid, detection, capabilities, pipeline)
    """

    def __init__(self, config: dict, output_dir: str = "reports"):
        viz = config.get("visualization", {})
        self.colormap   = viz.get("colormap", "RdYlBu_r")
        self.dpi        = viz.get("dpi", 150)
        self.figsize    = viz.get("figure_size", [14, 9])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        grid: Optional[BaselinedGrid],
        detection: Optional[DetectionResult],
        capabilities: DeviceCapabilities,
        pipeline: ComposedPipeline,
        output_prefix: str = "scan",
        raw_signals: Optional[np.ndarray] = None,
    ) -> dict[str, str]:
        """
        Main render dispatcher. Returns dict of output file paths.
        """
        output_paths: dict[str, str] = {}

        if capabilities.can_render_heatmap and grid is not None:
            p = self._render_heatmap(grid, detection, capabilities, pipeline, output_prefix)
            output_paths["heatmap"] = str(p)
        elif raw_signals is not None:
            p = self._render_line(raw_signals, capabilities, pipeline, output_prefix)
            output_paths["line_chart"] = str(p)
            logger.info(
                f"[VizEngine] No position data — rendered line chart: {p.name}"
            )
        else:
            logger.warning("[VizEngine] No data available for visualization.")

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    # 2D Scientific Heatmap
    # ─────────────────────────────────────────────────────────────────────────

    def _render_heatmap(
        self,
        grid: BaselinedGrid,
        detection: Optional[DetectionResult],
        cap: DeviceCapabilities,
        pipeline: ComposedPipeline,
        prefix: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#0d0d0d")
        ax.set_facecolor("#0d0d0d")

        gz = grid.grid_z.copy().astype(float)
        gz[~grid.grid_mask] = np.nan

        vmin = np.nanpercentile(gz, 2)
        vmax = np.nanpercentile(gz, 98)

        # Main heatmap
        im = ax.imshow(
            gz,
            origin="lower",
            extent=[
                grid.grid_x.min(), grid.grid_x.max(),
                grid.grid_y.min(), grid.grid_y.max(),
            ],
            cmap=self.colormap,
            vmin=vmin, vmax=vmax,
            aspect="equal",
            interpolation="bilinear",
        )

        # Anomaly contours
        try:
            ax.contour(
                grid.grid_x, grid.grid_y, gz,
                levels=8, colors="white", alpha=0.15, linewidths=0.5,
            )
        except Exception:
            pass

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Signal (normalized)", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Anomaly markers and dig zones
        if detection and detection.anomalies:
            self._draw_anomaly_markers(ax, detection, cap, grid)

        # Scan grid overlay
        self._draw_scan_grid(ax, grid)

        # Axes styling
        ax.set_xlabel("X (m)", color="white", fontsize=9)
        ax.set_ylabel("Y (m)", color="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Title
        grade_color, grade_label = GRADE_BADGE[cap.grade]
        title = f"GMS v2.3 — {prefix}"
        ax.set_title(title, color="white", fontsize=11, pad=10)

        # Telemetry grade badge
        ax.text(
            0.01, 0.99, f"● {grade_label}",
            transform=ax.transAxes, fontsize=8, color=grade_color,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111", edgecolor=grade_color, alpha=0.85),
        )

        # Disabled-features panel
        self._draw_disabled_panel(ax, cap, pipeline)

        # Noise floor annotation
        if grid.noise_floor > 0:
            ax.text(
                0.99, 0.01,
                f"Noise floor: {grid.noise_floor:.3f}",
                transform=ax.transAxes, fontsize=7, color="#888888",
                va="bottom", ha="right",
            )

        plt.tight_layout()
        out = self.output_dir / f"{prefix}_heatmap.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"[VizEngine] Heatmap saved: {out.name}")
        return out

    def _draw_anomaly_markers(
        self,
        ax: plt.Axes,
        detection: DetectionResult,
        cap: DeviceCapabilities,
        grid: BaselinedGrid,
    ):
        xmin, xmax = grid.grid_x.min(), grid.grid_x.max()
        ymin, ymax = grid.grid_y.min(), grid.grid_y.max()
        xspan = xmax - xmin or 1.0
        yspan = ymax - ymin or 1.0

        for i, anm in enumerate(detection.anomalies):
            if anm.raw_label == "NOISE":
                continue

            # Convert grid indices → world coordinates
            cx_w = xmin + (anm.marker_cx / (grid.grid_z.shape[1] - 1)) * xspan
            cy_w = ymin + (anm.marker_cy / (grid.grid_z.shape[0] - 1)) * yspan

            color = LABEL_COLORS.get(anm.raw_label, "#CCCCCC")
            sq_color = DIG_SQUARE_COLORS.get(anm.raw_label, "#FFFFFF")

            # DIG zone square
            half = max(0.02 * xspan, 0.3)
            rect = mpatches.FancyBboxPatch(
                (cx_w - half, cy_w - half), 2 * half, 2 * half,
                boxstyle="square,pad=0.02",
                linewidth=1.8, edgecolor=sq_color, facecolor="none",
                linestyle="--", zorder=5,
            )
            ax.add_patch(rect)

            # Crosshair
            ax.plot([cx_w - half * 0.6, cx_w + half * 0.6], [cy_w, cy_w],
                    color=sq_color, lw=0.8, zorder=6, alpha=0.8)
            ax.plot([cx_w, cx_w], [cy_w - half * 0.6, cy_w + half * 0.6],
                    color=sq_color, lw=0.8, zorder=6, alpha=0.8)

            # Confidence ring (only if SNR available)
            if cap.can_compute_confidence and anm.confidence > 0:
                radius = half * (1 + (1 - anm.confidence))
                ring = mpatches.Circle(
                    (cx_w, cy_w), radius,
                    fill=False, edgecolor=color, linestyle=":",
                    linewidth=1.0, alpha=0.6, zorder=4,
                )
                ax.add_patch(ring)

            # Uncertainty radius (only if SNR available)
            if cap.can_compute_uncertainty_radius and anm.uncertainty > 0:
                u_radius = anm.uncertainty * xspan * 0.05
                u_ring = mpatches.Circle(
                    (cx_w, cy_w), u_radius,
                    fill=False, edgecolor="#FFAA00", linestyle="-.",
                    linewidth=0.7, alpha=0.4, zorder=4,
                )
                ax.add_patch(u_ring)

            # Anomaly label
            label_text = f"T{i+1}: {anm.raw_label.replace('_', ' ')}"
            conf_text = (
                f" ({anm.confidence:.0%})"
                if cap.can_compute_confidence else ""
            )
            ax.text(
                cx_w, cy_w + half + 0.04 * yspan,
                label_text + conf_text,
                fontsize=7, color="white", ha="center", va="bottom",
                zorder=7,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )
            ax.text(
                cx_w, cy_w - half - 0.04 * yspan,
                "ESTIMATED POSITION ONLY",
                fontsize=5.5, color="#FFCC00", ha="center", va="top",
                zorder=7, alpha=0.8,
            )

    def _draw_scan_grid(self, ax: plt.Axes, grid: BaselinedGrid):
        """Draw faint scan grid lines."""
        for y in grid.grid_y[::max(1, len(grid.grid_y) // 10)]:
            ax.axhline(y, color="#333333", lw=0.3, alpha=0.5, zorder=1)
        for x in grid.grid_x[::max(1, len(grid.grid_x) // 10)]:
            ax.axvline(x, color="#333333", lw=0.3, alpha=0.5, zorder=1)

    def _draw_disabled_panel(
        self, ax: plt.Axes, cap: DeviceCapabilities, pipeline: ComposedPipeline
    ):
        """Show small annotations for disabled features."""
        disabled_msgs = []
        if not cap.can_compute_confidence:
            disabled_msgs.append("⚠ Confidence unavailable (no SNR)")
        if not cap.can_orientation_correct:
            disabled_msgs.append("⚠ Orientation correction unavailable (no heading)")
        if not cap.can_depth_estimate:
            disabled_msgs.append("⚠ Depth estimation disabled")

        if disabled_msgs:
            msg = "\n".join(disabled_msgs)
            ax.text(
                0.99, 0.99, msg,
                transform=ax.transAxes, fontsize=6.5, color="#AA6600",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                          edgecolor="#AA6600", alpha=0.75),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Line Visualization (fallback for no-position devices)
    # ─────────────────────────────────────────────────────────────────────────

    def _render_line(
        self,
        signals: np.ndarray,
        cap: DeviceCapabilities,
        pipeline: ComposedPipeline,
        prefix: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#0d0d0d")
        ax.set_facecolor("#111111")

        x = np.arange(len(signals))
        ax.plot(x, signals, color="#44AAFF", lw=1.0, alpha=0.9, label="Signal")

        # Rolling median baseline
        window = max(1, len(signals) // 20)
        baseline = np.array(pd.Series(signals).rolling(window, center=True, min_periods=1).median())
        ax.plot(x, baseline, color="#FF8844", lw=1.2, linestyle="--",
                alpha=0.7, label="Rolling baseline")

        ax.fill_between(x, signals, baseline,
                        where=signals > baseline, alpha=0.15, color="#44AAFF")
        ax.fill_between(x, signals, baseline,
                        where=signals < baseline, alpha=0.15, color="#FF4444")

        ax.set_xlabel("Sample index", color="white", fontsize=9)
        ax.set_ylabel("Signal", color="white", fontsize=9)
        ax.set_title(f"GMS v2.3 — {prefix} (Line Mode — no position)", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#222222", labelcolor="white", fontsize=8)

        grade_color, grade_label = GRADE_BADGE[cap.grade]
        ax.text(
            0.01, 0.99, f"● {grade_label}",
            transform=ax.transAxes, fontsize=8, color=grade_color,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                      edgecolor=grade_color, alpha=0.85),
        )
        ax.text(
            0.99, 0.99,
            "⚠ 2D heatmap unavailable: no x/y position in telemetry",
            transform=ax.transAxes, fontsize=7, color="#AA6600",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                      edgecolor="#AA6600", alpha=0.75),
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        plt.tight_layout()
        out = self.output_dir / f"{prefix}_line.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return out


# ── pandas import for rolling baseline in line view
import pandas as pd
