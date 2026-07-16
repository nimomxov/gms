"""
GMS — Interactive 3D Map Plugin  v2.2

Renders a fully interactive 3D geophysical surface with:
  - Rotatable/zoomable magnetic signal surface
  - Scan line grid overlay
  - Dig zone markers
  - Layer toggles (raw / filtered / anomaly mask / confidence)
  - Export: PNG + HTML (Plotly)

Uses matplotlib for PNG export (always available).
Uses Plotly for interactive HTML (optional, falls back gracefully).

The 3D view is physically meaningful:
  - XY plane = ground surface (scan coordinates)
  - Z axis = signal amplitude (NOT depth — see DepthInversionPlugin)
  - Color = signal intensity
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

logger = logging.getLogger("gms.viz3d")


class Interactive3DMapPlugin:
    """
    3D surface visualization of processed geophysical data.

    PNG output: matplotlib 3D surface (always available)
    HTML output: Plotly interactive surface (requires plotly)

    Layers available:
      "signal"     — filtered signal surface (default)
      "raw"        — unprocessed values
      "anomaly"    — binary anomaly mask
      "confidence" — per-cell confidence estimate

    Usage:
        plugin = Interactive3DMapPlugin(config)
        paths = plugin.render(grid, detection, final_report,
                               output_prefix="session_final",
                               layers=["signal", "anomaly"])
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("visualization", {})
        self.dpi     = cfg.get("dpi", 150)
        self.cmap    = cfg.get("colormap", "RdYlBu_r")
        self.figsize = cfg.get("figure_size", [14, 9])

    def render(self,
               grid,
               detection_result,
               final_report,
               output_prefix: str,
               output_dir: str = "reports",
               layers: list = None) -> dict:
        """
        Render all requested layers.
        Returns dict of {layer: path}.
        """
        layers  = layers or ["signal"]
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        paths   = {}

        for layer in layers:
            try:
                png_path = self._render_png(
                    grid, detection_result, final_report,
                    layer=layer, output_prefix=output_prefix, out_dir=out_dir
                )
                paths[f"{layer}_png"] = str(png_path)
            except Exception as e:
                logger.warning(f"  3D PNG layer={layer} failed: {e}")

        # Try Plotly HTML
        try:
            html_path = self._render_html(
                grid, detection_result, final_report,
                output_prefix=output_prefix, out_dir=out_dir
            )
            if html_path:
                paths["interactive_html"] = str(html_path)
        except Exception as e:
            logger.warning(f"  3D HTML failed: {e}")

        return paths

    # ── PNG (matplotlib) ──────────────────────────────────────────────────────

    def _render_png(self, grid, detection, final_report,
                    layer, output_prefix, out_dir) -> Path:
        png_path = out_dir / f"{output_prefix}_3D_{layer}.png"

        gz   = grid.grid_z.copy()
        mask = grid.grid_mask
        gz[~mask] = np.nan

        X, Y = np.meshgrid(grid.grid_x, grid.grid_y)

        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1]))
        ax  = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#0D0D0D")
        ax.set_facecolor("#0D0D0D")

        if layer == "signal":
            Z      = gz
            title  = "Signal Surface"
            cmap   = self.cmap
        elif layer == "anomaly":
            Z      = self._build_anomaly_mask(gz, detection)
            title  = "Anomaly Mask"
            cmap   = "hot"
        elif layer == "confidence":
            Z      = self._build_confidence_map(gz, detection)
            title  = "Confidence Map"
            cmap   = "YlOrRd"
        else:
            Z, title, cmap = gz, "Signal Surface", self.cmap

        # Surface plot
        surf = ax.plot_surface(
            X, Y, np.nan_to_num(Z, nan=0.0),
            cmap=cmap, alpha=0.85,
            linewidth=0, antialiased=True,
            rcount=60, ccount=60,
        )
        fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10,
                     label="Amplitude", pad=0.1)

        # Scan line grid overlay (thin lines at z=min)
        z_floor = float(np.nanmin(Z)) - float(np.nanstd(Z)) * 0.5
        stride  = max(1, len(grid.grid_y) // 10)
        for i in range(0, len(grid.grid_y), stride):
            ax.plot(grid.grid_x,
                    np.full_like(grid.grid_x, grid.grid_y[i]),
                    np.full_like(grid.grid_x, z_floor),
                    color="#444444", lw=0.4, alpha=0.5, zorder=1)

        # Target markers
        self._plot_targets_3d(ax, detection, final_report,
                              grid.grid_x, grid.grid_y, Z, z_floor)

        # Styling
        ax.set_xlabel("X (m)", color="#CCCCCC", labelpad=8)
        ax.set_ylabel("Y (m)", color="#CCCCCC", labelpad=8)
        ax.set_zlabel("Amplitude", color="#CCCCCC", labelpad=8)
        ax.tick_params(colors="#AAAAAA", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.grid(True, color="#333333", alpha=0.3)

        ax.set_title(f"GMS 3D {title} — {grid.scan_id}",
                     color="white", fontsize=10, pad=10)

        # Initial view angle
        ax.view_init(elev=30, azim=-60)

        fig.tight_layout()
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"  3D PNG ({layer}): {png_path}")
        return png_path

    def _plot_targets_3d(self, ax, detection, final_report,
                          grid_x, grid_y, Z, z_floor):
        """Plot anomaly markers at their true XY position, elevated above surface."""
        COLORS = {
            "FERROUS_METAL": "#FF4444",
            "CAVITY":        "#4488FF",
            "ROCK_DEBRIS":   "#FFAA00",
        }
        z_top = float(np.nanmax(Z)) * 1.05

        for a in detection.anomalies:
            if a.raw_label == "NOISE":
                continue
            mx = float(grid_x[min(int(a.marker_cx), len(grid_x)-1)])
            my = float(grid_y[min(int(a.marker_cy), len(grid_y)-1)])
            color = COLORS.get(a.raw_label, "#AAAAAA")
            ax.scatter([mx], [my], [z_top],
                       c=color, s=60*(0.5+a.confidence),
                       marker="^", edgecolors="white", lw=0.5, zorder=10)
            ax.plot([mx, mx], [my, my], [z_floor, z_top],
                    color=color, lw=0.6, alpha=0.4, zorder=9)

        # Confirmed DIG ZONE columns
        for ca in final_report.confirmed_anomalies:
            cx = float(grid_x[min(int(ca.centroid_x), len(grid_x)-1)])
            cy = float(grid_y[min(int(ca.centroid_y), len(grid_y)-1)])
            sq_color = "#FFFFFF" if ca.best_label == "FERROUS_METAL" else "#44AAFF"
            # Vertical post at dig zone
            ax.plot([cx, cx], [cy, cy], [z_floor, z_top*1.1],
                    color=sq_color, lw=2.0, alpha=0.9, zorder=11)
            ax.text(cx, cy, z_top*1.15,
                    f"DIG\n{ca.best_label[:3]}\n{ca.combined_confidence:.0%}",
                    color=sq_color, fontsize=6, ha="center", va="bottom",
                    zorder=12)

    def _build_anomaly_mask(self, gz, detection) -> np.ndarray:
        """Build a 2D confidence map from detected anomalies."""
        from scipy import ndimage as nd
        mask_z = np.zeros_like(gz)
        for a in detection.anomalies:
            if a.raw_label == "NOISE":
                continue
            r, c = int(a.cy), int(a.cx)
            if 0 <= r < gz.shape[0] and 0 <= c < gz.shape[1]:
                mask_z[r, c] = a.confidence
        # Spread
        mask_z = nd.gaussian_filter(mask_z, sigma=2.0)
        mask_z = np.where(np.isnan(gz), np.nan, mask_z)
        return mask_z

    def _build_confidence_map(self, gz, detection) -> np.ndarray:
        """Smoothed confidence map."""
        return self._build_anomaly_mask(gz, detection)

    # ── HTML (Plotly) ─────────────────────────────────────────────────────────

    def _render_html(self, grid, detection, final_report,
                     output_prefix, out_dir) -> str | None:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.debug("  Plotly not installed — skipping interactive HTML")
            return None

        gz   = grid.grid_z.copy()
        mask = grid.grid_mask
        gz[~mask] = np.nan

        fig = go.Figure()

        # ── Layer 1: Signal surface ───────────────────────────────────────────
        fig.add_trace(go.Surface(
            z=gz,
            x=grid.grid_x,
            y=grid.grid_y,
            colorscale="RdYlBu",
            reversescale=True,
            colorbar=dict(title="Amplitude", x=1.0),
            opacity=0.88,
            name="Signal Surface",
            visible=True,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>amp=%{z:.1f}<extra></extra>",
        ))

        # ── Layer 2: Anomaly mask (hidden by default) ─────────────────────────
        confidence_map = self._build_anomaly_mask(gz, detection)
        fig.add_trace(go.Surface(
            z=np.nan_to_num(confidence_map),
            x=grid.grid_x,
            y=grid.grid_y,
            colorscale="Hot",
            colorbar=dict(title="Confidence", x=1.1),
            opacity=0.60,
            name="Confidence Map",
            visible=False,
        ))

        # ── Scan grid lines ───────────────────────────────────────────────────
        z_floor = float(np.nanmin(gz)) - float(np.nanstd(gz)) * 0.3
        stride  = max(1, len(grid.grid_y) // 8)
        for i in range(0, len(grid.grid_y), stride):
            fig.add_trace(go.Scatter3d(
                x=grid.grid_x,
                y=np.full(len(grid.grid_x), grid.grid_y[i]),
                z=np.full(len(grid.grid_x), z_floor),
                mode="lines",
                line=dict(color="#444444", width=1),
                name=f"Scan line y={grid.grid_y[i]:.1f}",
                visible=True,
                showlegend=False,
            ))

        # ── Anomaly markers ───────────────────────────────────────────────────
        MARKER_COLORS = {
            "FERROUS_METAL": "#FF4444",
            "CAVITY":        "#4488FF",
            "ROCK_DEBRIS":   "#FFAA00",
        }
        z_top = float(np.nanmax(gz)) * 1.08

        for a in detection.anomalies:
            if a.raw_label == "NOISE":
                continue
            mx = float(grid.grid_x[min(int(a.marker_cx), len(grid.grid_x)-1)])
            my = float(grid.grid_y[min(int(a.marker_cy), len(grid.grid_y)-1)])

            fig.add_trace(go.Scatter3d(
                x=[mx], y=[my], z=[z_top],
                mode="markers+text",
                marker=dict(
                    size=8 + int(a.confidence * 6),
                    color=MARKER_COLORS.get(a.raw_label, "#888"),
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                text=[f"{a.raw_label}<br>conf:{a.confidence:.2f}"],
                textfont=dict(size=9, color="white"),
                name=a.raw_label,
                hovertemplate=(
                    f"<b>{a.raw_label}</b><br>"
                    f"x={mx:.2f} y={my:.2f}<br>"
                    f"confidence={a.confidence:.2f}<br>"
                    f"snr={a.snr_robust:.1f}<br>"
                    f"dipole={a.dipole_score:.3f}<br>"
                    "<i>Estimated position only</i><extra></extra>"
                ),
            ))

        # ── Confirmed DIG ZONE pillars ────────────────────────────────────────
        for ca in final_report.confirmed_anomalies:
            cx = float(grid.grid_x[min(int(ca.centroid_x), len(grid.grid_x)-1)])
            cy = float(grid.grid_y[min(int(ca.centroid_y), len(grid.grid_y)-1)])
            pillar_color = "#FFFFFF" if ca.best_label == "FERROUS_METAL" else "#44AAFF"

            fig.add_trace(go.Scatter3d(
                x=[cx, cx], y=[cy, cy], z=[z_floor, z_top*1.1],
                mode="lines+text",
                line=dict(color=pillar_color, width=4),
                text=["", f"<b>DIG</b><br>{ca.best_label}<br>{ca.combined_confidence:.0%}"],
                textfont=dict(size=10, color=pillar_color),
                name=f"DIG: {ca.best_label}",
                hovertemplate=(
                    f"<b>DIG ZONE</b><br>"
                    f"Label: {ca.best_label}<br>"
                    f"Confidence: {ca.combined_confidence:.0%}<br>"
                    f"Scans confirmed: {ca.scan_confirmations}<br>"
                    "<i>ESTIMATED POSITION — field verification required</i>"
                    "<extra></extra>"
                ),
            ))

        # ── Layout ────────────────────────────────────────────────────────────
        decision = final_report.decision
        dc = {"DIG": "#FF4444", "RESCAN": "#FFA500", "NO_DIG": "#44AA44"}

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>GMS 3D Analysis — {grid.scan_id}</b><br>"
                    f"<span style='color:{dc.get(decision,'#888')};font-size:14px'>"
                    f"Decision: {decision}</span> | "
                    f"Confirmed targets: {len(final_report.confirmed_anomalies)}"
                ),
                font=dict(color="white", size=13),
            ),
            scene=dict(
                xaxis=dict(title="X (m)", gridcolor="#333", showbackground=False),
                yaxis=dict(title="Y (m)", gridcolor="#333", showbackground=False),
                zaxis=dict(title="Amplitude", gridcolor="#333", showbackground=False),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
                bgcolor="#0D0D0D",
            ),
            paper_bgcolor="#0D0D0D",
            plot_bgcolor="#0D0D0D",
            font=dict(color="#CCCCCC"),
            height=700,
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.5, y=1.12, xanchor="center",
                buttons=[
                    dict(label="Signal", method="update",
                         args=[{"visible": self._layer_visibility("signal", fig)}]),
                    dict(label="Confidence", method="update",
                         args=[{"visible": self._layer_visibility("confidence", fig)}]),
                    dict(label="Both", method="update",
                         args=[{"visible": [True]*len(fig.data)}]),
                ],
                bgcolor="#1A1A1A", font=dict(color="white"),
                bordercolor="#555",
            )],
            annotations=[dict(
                text=(
                    "⚠ Estimated positions only — no depth information. "
                    "Requires field verification."
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.05, showarrow=False,
                font=dict(size=9, color="#888888"),
            )],
        )

        html_path = out_dir / f"{output_prefix}_3D_interactive.html"
        fig.write_html(str(html_path))
        logger.info(f"  3D HTML: {html_path}")
        return str(html_path)

    def _layer_visibility(self, layer: str, fig) -> list[bool]:
        """Build visibility list for Plotly layer toggle."""
        vis = []
        for trace in fig.data:
            name = trace.name or ""
            if layer == "signal":
                vis.append("Confidence" not in name)
            elif layer == "confidence":
                vis.append("Signal Surface" not in name)
            else:
                vis.append(True)
        return vis
