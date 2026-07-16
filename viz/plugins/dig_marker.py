"""
GMS — Dig Marker Visualization Plugin  v2.2

Shows the user EXACTLY WHERE TO DIG.

Design requirements:
  - White square around the TRUE dig zone (marker_cx, marker_cy)
  - NOT the blob centroid
  - Square size ∝ anomaly extent
  - Confidence text + target class + scan confirmation count
  - Dashed dipole connector line
  - Uncertainty radius circle
  - Cavity zones use a different marker (dashed blue rectangle)
  - Clean, unambiguous, field-usable visualization

Uses the validated dipole midpoint (marker_cx, marker_cy) computed
during detection, which is the geometric midpoint between the positive
and negative lobes of the magnetic dipole.

Scientific honesty:
  - Always shows uncertainty radius
  - Always shows scan confirmation count
  - Never shows depth (until calibrated inversion is available)
  - Labels "ESTIMATED POSITION" not "CONFIRMED POSITION"
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

logger = logging.getLogger("gms.dig_marker")

# ── Colour scheme ─────────────────────────────────────────────────────────────
TARGET_COLORS = {
    "FERROUS_METAL":  ("#FF2222", "#FFFFFF"),   # (fill, square_edge)
    "CAVITY":         ("#2266FF", "#00CCFF"),
    "ROCK_DEBRIS":    ("#CC8800", "#FFFFFF"),
    "SOIL_VARIATION": ("#6B8E23", "#CCCCCC"),
    "NOISE":          ("#888888", "#CCCCCC"),
}

DIG_SQUARE_COLORS = {
    "FERROUS_METAL": "#FFFFFF",
    "CAVITY":        "#44AAFF",
    "ROCK_DEBRIS":   "#FFCC44",
}


class DigMarkerPlugin:
    """
    Renders a professional field-usable dig-marker overlay on a heatmap.

    Call render_dig_overlay() after the base heatmap is drawn.
    Or use render_standalone_dig_map() for a dedicated dig-zone map.
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("visualization", {})
        self.dpi        = cfg.get("dpi", 150)
        self.figsize    = cfg.get("figure_size", [14, 9])
        self.marker_size = cfg.get("marker_size", 120)

    # ── Main overlay renderer ─────────────────────────────────────────────────

    def render_dig_overlay(self, ax: plt.Axes,
                            anomalies: list,
                            grid_x: np.ndarray,
                            grid_y: np.ndarray,
                            confirmed_groups: list = None) -> None:
        """
        Overlay dig markers on an existing matplotlib axis.

        Args:
            ax: matplotlib Axes with heatmap already drawn
            anomalies: list of RawAnomaly or Anomaly objects
            grid_x, grid_y: coordinate arrays for converting grid index → coords
            confirmed_groups: list of ConfirmedAnomaly (cross-scan validated)
        """
        # Draw per-scan anomalies (lighter)
        for a in anomalies:
            if a.raw_label == "NOISE":
                continue
            self._draw_single_anomaly(ax, a, grid_x, grid_y, alpha=0.65)

        # Draw confirmed cross-scan groups (bold, with dig squares)
        if confirmed_groups:
            for ca in confirmed_groups:
                self._draw_confirmed_target(ax, ca, grid_x, grid_y)

    def _grid_to_coord(self, idx: float, axis: np.ndarray) -> float:
        """Convert grid index (float) to coordinate."""
        i = int(np.clip(round(idx), 0, len(axis) - 1))
        return float(axis[i])

    def _draw_single_anomaly(self, ax, anomaly, grid_x, grid_y, alpha=0.65):
        """Draw a single anomaly marker at the TRUE position (marker_cx/cy)."""
        mx = self._grid_to_coord(anomaly.marker_cx, grid_x)
        my = self._grid_to_coord(anomaly.marker_cy, grid_y)
        cx = self._grid_to_coord(anomaly.cx, grid_x)
        cy = self._grid_to_coord(anomaly.cy, grid_y)

        fill_color, _ = TARGET_COLORS.get(anomaly.raw_label, ("#888", "#FFF"))

        marker_map = {
            "FERROUS_METAL": "^", "CAVITY": "o",
            "ROCK_DEBRIS": "s",   "SOIL_VARIATION": "D",
        }
        mkr = marker_map.get(anomaly.raw_label, "o")

        ax.scatter(mx, my,
                   s=self.marker_size * (0.5 + anomaly.confidence),
                   c=fill_color, marker=mkr,
                   edgecolors="white", linewidths=0.9,
                   zorder=6, alpha=alpha)

        # Dashed line: blob centroid → true marker position (dipole connector)
        if anomaly.raw_label == "FERROUS_METAL" and (mx != cx or my != cy):
            ax.plot([cx, mx], [cy, my],
                    color=fill_color, linewidth=0.8,
                    linestyle="--", alpha=0.5, zorder=5)

        # Label
        ax.annotate(
            f"{anomaly.raw_label[:3]} {anomaly.confidence:.2f}",
            (mx, my), fontsize=5.5, ha="center", va="bottom",
            color="white", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=fill_color,
                      alpha=0.7, linewidth=0),
        )

    def _draw_confirmed_target(self, ax, confirmed_anomaly, grid_x, grid_y):
        """
        Draw a bold DIG ZONE box for a cross-scan confirmed target.
        Uses the confirmed anomaly centroid (averaged across scans).
        """
        ca = confirmed_anomaly
        label  = ca.best_label
        conf   = ca.combined_confidence
        n_conf = ca.scan_confirmations

        # Convert centroid to coordinates
        # ConfirmedAnomaly stores centroid in grid-index space
        mx = self._grid_to_coord(ca.centroid_x, grid_x)
        my = self._grid_to_coord(ca.centroid_y, grid_y)

        sq_color = DIG_SQUARE_COLORS.get(label, "#FFFFFF")
        fill_col, _ = TARGET_COLORS.get(label, ("#888", "#FFF"))

        # Square size: proportional to confidence + label
        cell_size = float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 0.1
        if label == "CAVITY":
            half_w = cell_size * 12 * conf
            half_h = cell_size * 12 * conf
        else:
            half_w = cell_size * 8 * conf
            half_h = cell_size * 8 * conf

        # White DIG ZONE rectangle
        rect = mpatches.FancyBboxPatch(
            (mx - half_w, my - half_h),
            2 * half_w, 2 * half_h,
            boxstyle="square,pad=0.0",
            linewidth=2.0, edgecolor=sq_color,
            facecolor="none", zorder=8, alpha=0.95,
            linestyle=("--" if label == "CAVITY" else "-"),
        )
        ax.add_patch(rect)

        # Corner crosshair at true target position
        cross_size = half_w * 0.4
        ax.plot([mx - cross_size, mx + cross_size], [my, my],
                color=sq_color, lw=1.5, zorder=9, alpha=0.9)
        ax.plot([mx, mx], [my - cross_size, my + cross_size],
                color=sq_color, lw=1.5, zorder=9, alpha=0.9)

        # Uncertainty circle (radius ∝ uncertainty)
        unc = ca.mean_uncertainty
        if unc > 0.05:
            unc_circle = plt.Circle(
                (mx, my), radius=half_w * (0.3 + unc),
                fill=False, edgecolor=sq_color,
                linewidth=0.8, linestyle=":", alpha=0.5, zorder=7
            )
            ax.add_patch(unc_circle)

        # Bold label with confirmation count
        label_text = (
            f"DIG ZONE\n"
            f"{label}\n"
            f"conf: {conf:.0%}\n"
            f"scans: {n_conf}✓\n"
            f"[ESTIMATED POSITION]"
        )
        ax.annotate(
            label_text,
            (mx + half_w * 1.05, my),
            fontsize=6.5, ha="left", va="center",
            color="white", fontweight="bold", zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=fill_col, alpha=0.85, linewidth=0.5,
                edgecolor=sq_color
            ),
        )

    # ── Standalone dig map ────────────────────────────────────────────────────

    def render_standalone_dig_map(self,
                                   grid,
                                   detection_result,
                                   final_report,
                                   output_prefix: str,
                                   output_dir: str = "reports") -> str:
        """
        Generate a dedicated, field-ready DIG MAP.

        Shows:
          - Heatmap background (desaturated)
          - All scan anomalies (faded)
          - Confirmed DIG ZONES (bold white/blue squares)
          - Legend with instructions
          - Scientific disclaimer
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        png_path = out_dir / f"{output_prefix}_DIG_MAP.png"

        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor("#0A0A0A")
        ax.set_facecolor("#0A0A0A")

        gz = grid.grid_z.copy()
        gz[~grid.grid_mask] = np.nan

        # Desaturated heatmap background
        im = ax.imshow(
            gz, origin="lower",
            extent=[grid.grid_x.min(), grid.grid_x.max(),
                    grid.grid_y.min(), grid.grid_y.max()],
            cmap="gray", aspect="auto", interpolation="bilinear",
            alpha=0.45, zorder=1,
        )

        # Draw all anomalies (faded)
        for a in detection_result.anomalies:
            if a.raw_label == "NOISE":
                continue
            self._draw_single_anomaly(ax, a, grid.grid_x, grid.grid_y, alpha=0.40)

        # Draw confirmed dig zones (bold)
        for ca in final_report.confirmed_anomalies:
            self._draw_confirmed_target(ax, ca, grid.grid_x, grid.grid_y)

        # Decision banner
        decision = final_report.decision
        banner_colors = {"DIG": "#FF2222", "RESCAN": "#FFA500", "NO_DIG": "#44AA44"}
        bc = banner_colors.get(decision, "#888888")

        ax.text(
            0.5, 1.02,
            f"DECISION: {decision}  |  "
            f"Confirmed targets: {len(final_report.confirmed_anomalies)}  |  "
            f"Confidence: {final_report.confidence_summary.get('overall', 0):.0%}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=bc,
        )

        # Disclaimer
        ax.text(
            0.5, -0.07,
            "⚠ ESTIMATED POSITIONS ONLY — No depth information. "
            "All positions require field verification by a qualified geophysicist.",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7, color="#AAAAAA", style="italic",
        )

        # Legend
        legend_elements = [
            Line2D([0], [0], color="#FFFFFF", lw=2, ls="-",
                   label="Ferrous Metal DIG ZONE"),
            Line2D([0], [0], color="#44AAFF", lw=2, ls="--",
                   label="Cavity DIG ZONE"),
            Line2D([0], [0], color="#888888", lw=1, ls=":",
                   label="Uncertainty radius"),
            mpatches.Patch(facecolor="#FF2222", label="Ferrous metal"),
            mpatches.Patch(facecolor="#2266FF", label="Cavity / void"),
            mpatches.Patch(facecolor="#CC8800", label="Rock / debris"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                  fontsize=7, facecolor="#1A1A1A", edgecolor="#444",
                  labelcolor="white", framealpha=0.85)

        ax.set_xlabel("X (m)", color="#CCCCCC")
        ax.set_ylabel("Y (m)", color="#CCCCCC")
        ax.tick_params(colors="#CCCCCC")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        ax.set_title(
            f"GMS Field Dig Map — {grid.scan_id}",
            color="white", fontsize=11, pad=12
        )

        fig.tight_layout()
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"  Dig map saved: {png_path}")
        return str(png_path)
