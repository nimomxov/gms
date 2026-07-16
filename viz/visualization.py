"""
GMS Visualization — Heatmap & Anomaly Markers
Exports PNG + optional interactive HTML (Plotly).
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib import cm

from core.signal_processing import ProcessedGrid
from core.anomaly_detection import DetectionResult
from core.decision_engine import FinalReport

logger = logging.getLogger("gms.viz")

LABEL_COLORS = {
    "FERROUS_METAL":  "#FF2222",   # RED   — metal
    "CAVITY":         "#2266FF",   # BLUE  — void / grave
    "ROCK_DEBRIS":    "#CC8800",   # AMBER — debris
    "SOIL_VARIATION": "#6B8E23",   # OLIVE — soil
    "NOISE":          "#AAAAAA",   # GREY  — rejected
    "CONFIRMED":      "#FF2222",   # RED star for confirmed metal (overridden by label below)
    "UNKNOWN":        "#CCCCCC",
}

LABEL_MARKERS = {
    "FERROUS_METAL":  "^",    # triangle up
    "CAVITY":         "o",    # circle
    "ROCK_DEBRIS":    "s",    # square
    "SOIL_VARIATION": "D",    # diamond
    "NOISE":          "x",
    "CONFIRMED":      "*",    # star for confirmed
}


class GeoVizEngine:
    """Generates geophysical heatmaps with annotated anomaly markers."""

    def __init__(self, config: dict, output_dir: str = "reports"):
        viz = config.get("visualization", {})
        self.colormap = viz.get("colormap", "RdYlBu_r")
        self.dpi = viz.get("dpi", 150)
        self.export_html = viz.get("export_html", True)
        self.marker_size = viz.get("marker_size", 120)
        self.figsize = viz.get("figure_size", [12, 8])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_scan_heatmap(self,
                             grid: ProcessedGrid,
                             detection: DetectionResult,
                             output_prefix: str = None) -> dict:
        """Render a single scan heatmap with detected anomalies."""
        prefix = output_prefix or grid.scan_id
        png_path = self.output_dir / f"{prefix}_heatmap.png"

        fig, ax = plt.subplots(figsize=self.figsize)

        gz = grid.grid_z.copy()
        gz[~grid.grid_mask] = np.nan

        vmin, vmax = np.nanpercentile(gz, 2), np.nanpercentile(gz, 98)
        im = ax.imshow(
            gz,
            origin="lower",
            extent=[grid.grid_x.min(), grid.grid_x.max(),
                    grid.grid_y.min(), grid.grid_y.max()],
            cmap=self.colormap,
            vmin=vmin, vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )

        plt.colorbar(im, ax=ax, label="Signal Amplitude (drift-removed)")

        # Plot anomalies
        for a in detection.anomalies:
            if a.raw_label == "NOISE":
                continue

            # Convert grid index → coordinate
            cx_coord  = grid.grid_x[min(int(a.cx),  len(grid.grid_x) - 1)]
            cy_coord  = grid.grid_y[min(int(a.cy),  len(grid.grid_y) - 1)]
            mx_coord  = grid.grid_x[min(int(a.marker_cx), len(grid.grid_x) - 1)]
            my_coord  = grid.grid_y[min(int(a.marker_cy), len(grid.grid_y) - 1)]

            color  = LABEL_COLORS.get(a.raw_label, "#CCCCCC")
            marker = LABEL_MARKERS.get(a.raw_label, "o")

            # For ferrous metal: draw a dashed line from blob centroid to midpoint
            # so the user can see both the dipole lobes and the true object location
            if a.raw_label == "FERROUS_METAL" and (cx_coord != mx_coord or cy_coord != my_coord):
                ax.plot([cx_coord, mx_coord], [cy_coord, my_coord],
                        color=color, linewidth=0.8, linestyle="--",
                        alpha=0.6, zorder=4)

            # Main marker at TRUE object position (midpoint for metal, centroid otherwise)
            ax.scatter(mx_coord, my_coord,
                       s=self.marker_size * (0.5 + a.confidence),
                       c=color, marker=marker,
                       edgecolors="black", linewidths=0.8,
                       zorder=5, alpha=0.90)

            ax.annotate(
                f"{a.raw_label[:3]}\n{a.confidence:.2f}",
                (mx_coord, my_coord), fontsize=6, ha="center", va="bottom",
                color="white",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.65)
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"GMS Scan: {grid.scan_id}\n"
            f"Noise floor: {grid.noise_floor:.2f}  "
            f"Dynamic range: {grid.dynamic_range:.2f}  "
            f"Quality: {detection.scan_quality_score:.2f}",
            fontsize=10
        )

        # Legend
        legend_handles = []
        seen_labels = set(a.raw_label for a in detection.anomalies if a.raw_label != "NOISE")
        for lbl in seen_labels:
            handle = mpatches.Patch(
                color=LABEL_COLORS.get(lbl, "#CCC"),
                label=lbl.replace("_", " ")
            )
            legend_handles.append(handle)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved heatmap: {png_path}")

        paths = {"png": str(png_path)}

        if self.export_html:
            html_path = self._render_interactive(grid, detection, prefix)
            if html_path:
                paths["html"] = str(html_path)

        return paths

    def render_final_report_map(self,
                                 grids: list[ProcessedGrid],
                                 report: FinalReport,
                                 output_prefix: str = "final") -> dict:
        """Multi-scan overview map showing confirmed anomalies."""
        png_path = self.output_dir / f"{output_prefix}_confirmed_anomalies.png"

        fig, axes = plt.subplots(
            1, max(1, len(grids)),
            figsize=(self.figsize[0] * len(grids), self.figsize[1]),
            squeeze=False
        )

        grid_map = {g.scan_id: g for g in grids}

        for col, grid in enumerate(grids):
            ax = axes[0][col]
            gz = grid.grid_z.copy()
            gz[~grid.grid_mask] = np.nan
            ax.imshow(gz, origin="lower", cmap=self.colormap, aspect="auto",
                      interpolation="bilinear")
            ax.set_title(f"Scan: {grid.scan_id}", fontsize=8)

            # Mark confirmed anomalies
            for ca in report.confirmed_anomalies:
                if any(s in grid.scan_id for s in ca.contributing_scans):
                    marker_color = LABEL_COLORS.get(ca.best_label, "#FF2222")
                    ax.scatter(
                        ca.centroid_x, ca.centroid_y,
                        s=200 * ca.combined_confidence,
                        c=marker_color,
                        marker="*", edgecolors="yellow", linewidths=1.2,
                        zorder=10, label=f"{ca.best_label} ({ca.combined_confidence:.2f})"
                    )

        decision_color = {"DIG": "#FF4444", "RESCAN": "#FFA500", "NO_DIG": "#44AA44"}
        fig.suptitle(
            f"GMS Final Decision: {report.decision}  |  "
            f"Confirmed Anomalies: {len(report.confirmed_anomalies)}  |  "
            f"Session: {report.session_id}",
            fontsize=12, fontweight="bold",
            color=decision_color.get(report.decision, "black")
        )

        fig.tight_layout()
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved final map: {png_path}")
        return {"png": str(png_path)}

    def _render_interactive(self, grid: ProcessedGrid,
                             detection: DetectionResult,
                             prefix: str) -> str | None:
        """Generate interactive Plotly HTML heatmap."""
        try:
            import plotly.graph_objects as go

            gz = grid.grid_z.copy()
            gz[~grid.grid_mask] = np.nan

            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=gz,
                x=grid.grid_x,
                y=grid.grid_y,
                colorscale="RdYlBu_r",
                colorbar=dict(title="Amplitude"),
                hoverongaps=False,
            ))

            for a in detection.anomalies:
                if a.raw_label in ("NOISE",):
                    continue
                ax_x = grid.grid_x[min(int(a.cx), len(grid.grid_x) - 1)]
                ax_y = grid.grid_y[min(int(a.cy), len(grid.grid_y) - 1)]
                color = LABEL_COLORS.get(a.raw_label, "#CCC")

                fig.add_trace(go.Scatter(
                    x=[ax_x], y=[ax_y],
                    mode="markers+text",
                    marker=dict(size=14, color=color, symbol="star",
                                line=dict(width=1, color="black")),
                    text=[f"{a.raw_label}<br>conf:{a.confidence:.2f}<br>SNR:{a.snr_robust:.1f}"],
                    textposition="top center",
                    name=a.raw_label,
                    hoverinfo="text",
                ))

            fig.update_layout(
                title=f"GMS Interactive Heatmap — {grid.scan_id}",
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                template="plotly_dark",
                height=600,
            )

            html_path = self.output_dir / f"{prefix}_heatmap.html"
            fig.write_html(str(html_path))
            logger.info(f"  Saved interactive HTML: {html_path}")
            return str(html_path)

        except ImportError:
            logger.warning("Plotly not installed — skipping interactive HTML export")
            return None
