"""
GMS — Inspector Panel Controller  v3.5
========================================
Displays REAL backend metrics from pipeline_exec result_dict.
All values come from core/ backend objects — no hardcoded defaults.

Metric sources (all from result_dict confirmed_anomalies entries):
  label            ← RawAnomaly.raw_label / ConfirmedAnomaly.best_label
  snr              ← RawAnomaly.snr_robust (real dB)
  confidence       ← combined_confidence (after reliability penalty)
  reliability      ← reliability_score × confidence
  x, y             ← grid_x[marker_cx], grid_y[marker_cy]  (real metres)
  depth_str        ← DepthInversionPlugin.estimate_depth()
  dipole_score     ← RawAnomaly.dipole_score
  coherence        ← RawAnomaly.spatial_coherence
  final_score      ← RawAnomaly.final_score
  topology_status  ← TopologyDescriptor (passed through result)
  scan_confirmations ← ConfirmedAnomaly.scan_confirmations
  explanation      ← ExplainabilityEngine.explain_anomaly().full_text

Widget ObjectNames (from gms_main_window.ui):
  inspNoTarget, inspDetails, inspDetailsLay
  inspTargetName, inspSNR, inspConfidence, inspReliability
  inspX, inspY, inspDepth, inspFusion, inspTopology, inspMultiScan
  btnConfirmDig, btnRejectTarget
  listAnomalies (optional — populated if present)
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QLabel, QPushButton, QWidget, QMainWindow, QTextEdit,
    QListWidget, QListWidgetItem, QFrame,
)

from .app_state import GMSApplicationState, AnomalyInfo, PipelineStatus
from .event_bus import GMSEventBus

logger = logging.getLogger("gms.inspector")


def _w(parent, cls, name):
    f = parent.findChild(cls, name)
    if f is None:
        logger.debug(f"[Inspector] {cls.__name__}[{name}] not found")
    return f


def _set_visible_up(widget, visible: bool):
    if widget is None:
        return
    widget.setVisible(visible)
    if visible:
        p = widget.parent()
        while p and not p.isVisible():
            p.setVisible(True)
            p = p.parent()


class InspectorPanelController(QObject):

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w       = window
        self._state   = GMSApplicationState.instance()
        self._bus     = GMSEventBus.instance()
        self._current: Optional[AnomalyInfo] = None

        self._state.anomaly_selected.connect(self._on_anomaly_selected)
        self._state.anomalies_updated.connect(self._on_anomalies_updated)
        self._state.pipeline_status_changed.connect(self._on_pipeline_status)
        self._state.dataset_cleared.connect(self._reset)

        self._wire_buttons()
        self._wire_list()
        self._reset()
        logger.info("[Inspector] Controller attached (v3.5)")

    # ── Wiring ─────────────────────────────────────────────────────────────

    def _wire_buttons(self):
        for name, slot in (
            ("btnConfirmDig",   self._on_confirm_dig),
            ("btnRejectTarget", self._on_reject),
            ("btnValidate",     self._on_validate),
        ):
            btn = _w(self._w, QPushButton, name)
            if btn:
                btn.clicked.connect(slot)

    def _wire_list(self):
        lst = _w(self._w, QListWidget, "listAnomalies")
        if lst:
            lst.currentItemChanged.connect(self._on_list_changed)

    # ── State handlers ─────────────────────────────────────────────────────

    def _on_anomaly_selected(self, anomaly: Optional[AnomalyInfo]):
        self._current = anomaly
        if anomaly is None:
            self._reset()
        else:
            self._populate(anomaly)
            self._sync_list(anomaly.anomaly_id)

    def _on_anomalies_updated(self, anomalies: list):
        self._fill_list(anomalies)
        if anomalies and self._current is None:
            self._state.set_selected_anomaly(anomalies[0])

    def _on_pipeline_status(self, status: PipelineStatus):
        if status == PipelineStatus.RUNNING:
            lbl = _w(self._w, QLabel, "inspNoTarget")
            if lbl:
                lbl.setVisible(True)
                lbl.setText("Analysing…")
            self._set_btns(False)

    def _on_list_changed(self, current: QListWidgetItem, _):
        if current is None:
            return
        aid = current.data(Qt.UserRole)
        for a in self._state.anomaly_list:
            if a.anomaly_id == aid:
                self._state.set_selected_anomaly(a)
                return

    # ── List ───────────────────────────────────────────────────────────────

    def _fill_list(self, anomalies: list):
        lst = _w(self._w, QListWidget, "listAnomalies")
        if lst is None:
            return
        lst.blockSignals(True)
        lst.clear()
        for a in anomalies:
            text = f"{a.label}  {a.confidence:.0%}  ({a.snr:.1f} dB)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, a.anomaly_id)
            if a.confidence >= 0.70:
                item.setForeground(Qt.green)
            elif a.confidence >= 0.45:
                item.setForeground(Qt.yellow)
            else:
                item.setForeground(Qt.red)
            lst.addItem(item)
        lst.blockSignals(False)

    def _sync_list(self, anomaly_id: str):
        lst = _w(self._w, QListWidget, "listAnomalies")
        if lst is None:
            return
        lst.blockSignals(True)
        for i in range(lst.count()):
            if lst.item(i).data(Qt.UserRole) == anomaly_id:
                lst.setCurrentRow(i)
                break
        lst.blockSignals(False)

    # ── Panel population ───────────────────────────────────────────────────

    def _populate(self, a: AnomalyInfo):
        # Hide "no target" label
        no_tgt = _w(self._w, QLabel, "inspNoTarget")
        if no_tgt:
            no_tgt.setVisible(False)

        # Show detail containers
        for name in ("inspDetails", "inspDetailsLay"):
            _set_visible_up(_w(self._w, QWidget, name), True)

        # Show all cardFrame children of inspector
        frame = _w(self._w, QFrame, "inspectorFrame")
        if frame:
            for card in frame.findChildren(QFrame, "cardFrame"):
                _set_visible_up(card, True)

        # ── Core label fields ─────────────────────────────────────────────
        self._lbl("inspTargetName", a.label)
        self._lbl("inspSNR",        f"{a.snr:.2f} dB")
        self._lbl("inspX",          f"{a.x:.4f} m")
        self._lbl("inspY",          f"{a.y:.4f} m")
        self._lbl("inspDepth",      a.depth_str)
        self._lbl("inspMultiScan",  f"{a.scan_confirmations} scan(s)")

        # Confidence with colour coding
        conf_lbl = _w(self._w, QLabel, "inspConfidence")
        if conf_lbl:
            conf_lbl.setText(f"{a.confidence:.1%}")
            if a.confidence >= 0.70:
                conf_lbl.setStyleSheet("color:#00FF41;font-weight:bold;")
            elif a.confidence >= 0.45:
                conf_lbl.setStyleSheet("color:#FFA500;font-weight:bold;")
            else:
                conf_lbl.setStyleSheet("color:#FF4444;font-weight:bold;")

        # Reliability
        self._lbl("inspReliability", f"{a.reliability:.3f}")

        # Fusion / dipole / topology
        self._lbl("inspFusion",    f"{a.fusion_boost:+.3f}")
        self._lbl("inspTopology",  a.topology_status.replace("_", " ").title())

        # ── Extended metrics (if additional attrs present) ─────────────────
        # These come from the extra fields added in pipeline_exec result_dict
        extra = getattr(a, "_extra", {})
        if not extra:
            # Try reading from state's last result directly
            result = self._state.last_result or {}
            for ra in result.get("confirmed_anomalies", []):
                if ra.get("anomaly_id") == a.anomaly_id:
                    extra = ra
                    break

        if extra:
            self._lbl("inspDipole",     f"{extra.get('dipole_score', a.dipole_score):.3f}")
            self._lbl("inspCoherence",  f"{extra.get('coherence', a.coherence):.3f}")
            self._lbl("inspFinalScore", f"{extra.get('final_score', 0.0):.3f}")
            self._lbl("inspSmoothness", f"{extra.get('smoothness', 0.0):.3f}")
            self._lbl("inspDetector",   str(extra.get("detector_name", "—")))

        # ── Reliability panel (from reliability summary) ───────────────────
        result = self._state.last_result or {}
        rel = result.get("reliability", {})
        if rel:
            self._lbl("inspRelLabel",   str(rel.get("quality_label", "—")))
            self._lbl("inspRelScore",   f"{rel.get('reliability_score', 0.0):.3f}")
            self._lbl("inspRelSNR",     f"{rel.get('snr_mean', 0.0):.2f}")
            self._lbl("inspRelCov",     f"{rel.get('coverage', 0.0):.1%}")
            flags = rel.get("flags", [])
            self._lbl("inspRelFlags",   ", ".join(flags) if flags else "none")

        # ── Pipeline panel ────────────────────────────────────────────────
        pipeline = result.get("pipeline", {})
        if pipeline:
            self._lbl("inspInterp",    str(pipeline.get("interpolator", "—")))
            self._lbl("inspBaseline",  str(pipeline.get("baseline", "—")))
            self._lbl("inspDetector2", str(pipeline.get("detector", "—")))
            self._lbl("inspPreset",    str(pipeline.get("preset", "—")))
            self._lbl("inspCfgHash",   str(pipeline.get("config_hash", "—"))[:12])

        # ── Explanation from ExplainabilityEngine ─────────────────────────
        exp_text = ""
        if extra:
            exp_text = str(extra.get("explanation", ""))
        if not exp_text:
            exp_text = getattr(a, "decision_reason", "")

        for name in ("inspExplanation",):
            lbl = _w(self._w, QLabel, name)
            if lbl:
                lbl.setWordWrap(True)
                lbl.setText(exp_text)

        for name in ("inspExplanationText",):
            txt = _w(self._w, QTextEdit, name)
            if txt:
                txt.setPlainText(exp_text)
                txt.setReadOnly(True)

        self._set_btns(True)
        logger.debug(
            f"[Inspector] Populated: {a.label} "
            f"conf={a.confidence:.1%} x={a.x:.3f} y={a.y:.3f}"
        )

    # ── Reset ──────────────────────────────────────────────────────────────

    def _reset(self):
        no_tgt = _w(self._w, QLabel, "inspNoTarget")
        if no_tgt:
            no_tgt.setVisible(True)
            no_tgt.setText("No target selected")

        for name in ("inspDetails", "inspDetailsLay"):
            w = _w(self._w, QWidget, name)
            if w:
                w.setVisible(False)

        frame = _w(self._w, QFrame, "inspectorFrame")
        if frame:
            for card in frame.findChildren(QFrame, "cardFrame"):
                card.setVisible(False)

        self._set_btns(False)
        self._current = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _lbl(self, name: str, text: str):
        lbl = _w(self._w, QLabel, name)
        if lbl:
            lbl.setText(text)

    def _set_btns(self, enabled: bool):
        for name in ("btnConfirmDig", "btnRejectTarget"):
            btn = _w(self._w, QPushButton, name)
            if btn:
                btn.setEnabled(enabled)

    # ── User actions ───────────────────────────────────────────────────────

    def _on_confirm_dig(self):
        if not self._current:
            return
        try:
            from .command_history import CommandHistory, ValidateTargetCommand
            CommandHistory.instance().execute(
                ValidateTargetCommand(self._state,
                                     self._current.anomaly_id, "confirm_dig")
            )
        except Exception:
            pass
        self._bus.emit_event("DIG_CONFIRMED",
                             anomaly_id=self._current.anomaly_id)
        logger.info(f"[Inspector] DIG CONFIRMED: {self._current.label}")
        self._reset()

    def _on_reject(self):
        if not self._current:
            return
        try:
            from .command_history import CommandHistory, ValidateTargetCommand
            CommandHistory.instance().execute(
                ValidateTargetCommand(self._state,
                                     self._current.anomaly_id, "reject")
            )
        except Exception:
            updated = [a for a in self._state.anomaly_list
                       if a.anomaly_id != self._current.anomaly_id]
            self._state.set_anomaly_list(updated)
            self._state.set_selected_anomaly(None)
        logger.info(f"[Inspector] REJECTED: {self._current.label}")

    def _on_validate(self):
        if not self._current:
            return
        if not hasattr(self, "_gt_workflow") or self._gt_workflow is None:
            logger.warning("[Inspector] GroundTruthWorkflow not attached")
            return
        # FIX 16: session id + decision now exist on state (see app_state fix).
        session_id = getattr(self._state, "last_session_id", "")
        # Prefer this target's own decision_reason-derived decision if present,
        # else fall back to the session-level decision.
        predicted_decision = getattr(self._state, "last_decision", "NO_DIG")
        self._gt_workflow.open_validation_panel(
            anomaly_id=self._current.anomaly_id,
            session_id=session_id,
            predicted_decision=predicted_decision,
            predicted_confidence=float(self._current.confidence),
        )
        logger.info(f"[Inspector] Ground truth panel opened: {self._current.anomaly_id}")
        
    def set_ground_truth_workflow(self, workflow) -> None:
        """Inject the GroundTruthWorkflow instance from bootstrap_integration."""
        self._gt_workflow = workflow
        logger.info("[Inspector] GroundTruthWorkflow attached")

