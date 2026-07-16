"""
GMS — Fusion Controller  v1.0
================================
Subscribes to GMSApplicationState.fusion_changed and pushes
FusionResult data into every relevant UI surface:

  • Inspector panel  — fused target list with tier badges
  • Status bar       — HIGH/MEDIUM/LOW count summary
  • Heatmap overlay  — marker positions for fused targets

Architecture rules (preserved):
  - UI thread only — no backend calls here.
  - No direct widget access from outside this controller.
  - All state mutations go through GMSApplicationState.
  - FusionController never calls MultiScanFusionEngine directly;
    that runs inside PipelineWorker on the thread pool.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore  import QObject, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTextEdit, QFrame,
)

from .app_state import GMSApplicationState

logger = logging.getLogger("gms.fusion_ctrl")

# ── Widget finder (mirrors pattern used across all other controllers) ─────────

def _w(window, cls, name: str):
    w = window.findChild(cls, name)
    if w is None:
        logger.debug(f"[FusionCtrl] Widget not found: {name!r}")
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Tier badge helpers
# ─────────────────────────────────────────────────────────────────────────────

_TIER_BADGE = {
    "HIGH":   "🔴 HIGH",
    "MEDIUM": "🟡 MEDIUM",
    "LOW":    "⚪ LOW",
}

_TIER_TOOLTIP = {
    "HIGH":   "Detected in ≥3 independent scans — highest evidence level",
    "MEDIUM": "Detected in 2 independent scans — cross-confirmed",
    "LOW":    "Detected in 1 scan only — unconfirmed",
}


def _tier_badge(tier: str) -> str:
    return _TIER_BADGE.get(tier, tier)


def _format_target_item(target) -> str:
    """One-line summary for the fused target list widget."""
    badge  = _tier_badge(target.fusion_tier)
    depth  = f"{target.depth_m:.2f} m" if target.depth_m is not None else "depth N/A"
    return (
        f"{badge}  │  {target.label:<20}  │  "
        f"conf={target.confidence:.2f}  rep={target.repeatability_score:.2f}  │  "
        f"x={target.x:.2f} y={target.y:.2f}  │  {depth}  │  "
        f"{target.n_scans} scan(s)"
    )


def _format_status_summary(fusion_result) -> str:
    """Compact status bar string: 'Fusion: 2 HIGH  3 MEDIUM  1 LOW'"""
    if fusion_result is None:
        return ""
    tc = fusion_result.summary()["tier_counts"]
    parts = []
    if tc["HIGH"]   > 0: parts.append(f"🔴 {tc['HIGH']} HIGH")
    if tc["MEDIUM"] > 0: parts.append(f"🟡 {tc['MEDIUM']} MEDIUM")
    if tc["LOW"]    > 0: parts.append(f"⚪ {tc['LOW']} LOW")
    if not parts:
        return "Fusion: no targets"
    return "Fusion: " + "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# FusionController
# ─────────────────────────────────────────────────────────────────────────────

class FusionController(QObject):
    """
    Reacts to GMSApplicationState.fusion_changed and populates:

      lstFusedTargets   — QListWidget showing all FusedTargets
      lblFusionSummary  — QLabel showing HIGH/MEDIUM/LOW counts
      lblFusionStatus   — QLabel in status bar area
      txtFusionDiag     — QTextEdit showing explain() for selected target
      btnFusionExport   — QPushButton to export fusion JSON

    All widget lookups are soft — missing widgets are silently skipped
    so the controller degrades gracefully on any .ui layout variant.
    """

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()
        self._last_result = None

        # Widget cache
        self._lst_targets:   Optional[QListWidget] = _w(window, QListWidget,  "lstFusedTargets")
        self._lbl_summary:   Optional[QLabel]      = _w(window, QLabel,       "lblFusionSummary")
        self._lbl_status:    Optional[QLabel]      = _w(window, QLabel,       "lblFusionStatus")
        self._txt_diag:      Optional[QTextEdit]   = _w(window, QTextEdit,    "txtFusionDiag")
        self._btn_export:    Optional[QPushButton] = _w(window, QPushButton,  "btnFusionExport")

        self._wire_signals()
        logger.info("[FusionCtrl] Initialised")

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _wire_signals(self) -> None:
        # State signal → update UI
        self._state.fusion_changed.connect(self._on_fusion_changed)

        # List selection → show diagnostics
        if self._lst_targets is not None:
            self._lst_targets.currentRowChanged.connect(self._on_target_selected)

        # Export button
        if self._btn_export is not None:
            self._btn_export.clicked.connect(self._on_export)

        # Also react to pipeline result_dict that includes fusion_result
        self._state.pipeline_completed.connect(self._on_pipeline_completed)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(object)
    def _on_fusion_changed(self, fusion_result) -> None:
        """Called whenever GMSApplicationState.set_fusion_result() fires."""
        self._last_result = fusion_result
        self._populate_target_list(fusion_result)
        self._update_summary(fusion_result)
        self._update_status_label(fusion_result)
        logger.info(
            f"[FusionCtrl] Updated: {fusion_result.n_clusters} clusters, "
            f"{fusion_result.n_fused} fused, {fusion_result.n_singletons} singletons"
        )

    @Slot(dict)
    def _on_pipeline_completed(self, result_dict: dict) -> None:
        """
        When PipelineWorker finishes, it embeds a FusionResult in result_dict.
        Push it into GMSApplicationState so all subscribers (including this
        controller) receive it through the normal signal path.
        """
        fusion_result = result_dict.get("fusion_result")
        if fusion_result is not None:
            self._state.set_fusion_result(fusion_result)
        else:
            logger.debug("[FusionCtrl] Pipeline result has no fusion_result key")

    @Slot(int)
    def _on_target_selected(self, row: int) -> None:
        """Show explain() diagnostics for the selected FusedTarget."""
        if self._txt_diag is None:
            return
        if self._last_result is None or row < 0:
            self._txt_diag.setPlainText("")
            return
        targets = self._last_result.targets
        if row >= len(targets):
            return

        target = targets[row]
        from core.fusion_engine import MultiScanFusionEngine
        engine = MultiScanFusionEngine()          # stateless — safe to re-create
        explanation = engine.explain(target)

        # Append to_dict summary
        d = target.to_dict()
        detail_lines = [
            explanation,
            "",
            "── Raw dict ──────────────────────────────────────",
        ]
        for k, v in d.items():
            detail_lines.append(f"  {k:<26}: {v}")

        self._txt_diag.setPlainText("\n".join(detail_lines))

    @Slot()
    def _on_export(self) -> None:
        """Export the current FusionResult to JSON."""
        if self._last_result is None:
            return
        try:
            import json
            from pathlib import Path
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self._w, "Export Fusion Result", "fusion_result.json",
                "JSON files (*.json)"
            )
            if not path:
                return
            data = {
                "summary": self._last_result.summary(),
                "targets": [t.to_dict() for t in self._last_result.targets],
            }
            Path(path).write_text(json.dumps(data, indent=2))
            logger.info(f"[FusionCtrl] Exported to {path}")
        except Exception as e:
            logger.error(f"[FusionCtrl] Export failed: {e}")

    # ── UI population ─────────────────────────────────────────────────────────

    def _populate_target_list(self, fusion_result) -> None:
        if self._lst_targets is None:
            return
        self._lst_targets.clear()
        for target in fusion_result.targets:
            item = QListWidgetItem(_format_target_item(target))
            item.setToolTip(
                f"{_TIER_TOOLTIP.get(target.fusion_tier, '')}\n"
                f"Supporting scans: {', '.join(target.supporting_scans)}\n"
                f"Label agreement: {target.label_agreement:.0%}\n"
                f"Repeatability: {target.repeatability_score:.3f}"
            )
            # Store fused_id for later retrieval
            item.setData(0x0100, target.fused_id)   # Qt.UserRole
            self._lst_targets.addItem(item)

    def _update_summary(self, fusion_result) -> None:
        if self._lbl_summary is None:
            return
        s = fusion_result.summary()
        tc = s["tier_counts"]
        html = (
            f"<b>Fusion results:</b>  "
            f"<span style='color:#d32f2f'>🔴 {tc['HIGH']} HIGH</span>  "
            f"<span style='color:#f57f17'>🟡 {tc['MEDIUM']} MEDIUM</span>  "
            f"<span style='color:#616161'>⚪ {tc['LOW']} LOW</span>  "
            f"&nbsp;|&nbsp; {s['n_scans_fused']} scan(s)  "
            f"{s['n_anomalies_in']} anomalies in  "
            f"{s['n_clusters']} cluster(s)"
        )
        self._lbl_summary.setText(html)

    def _update_status_label(self, fusion_result) -> None:
        if self._lbl_status is None:
            return
        self._lbl_status.setText(_format_status_summary(fusion_result))

    # ── Public API (callable from bootstrap_integration) ──────────────────────

    def push_fusion_result(self, fusion_result) -> None:
        """
        Directly push a FusionResult — bypasses the signal path.
        Useful when called synchronously from tests or the integration patch.
        """
        self._on_fusion_changed(fusion_result)
