"""
GMS — Target Possibility Controller v1.0
===========================================
Drives the Target Possibility toolbar button end-to-end:
  * runs TargetPossibilityEngine on the last pipeline result (no user input)
  * shows a staged progress dialog (one line per completed stage)
  * renders the professional per-target report
  * colors the map by possibility and offers the hide filters
Never mutates raw data; reads GMSApplicationState.last_result only.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThreadPool, QRunnable, Signal
from PySide6.QtWidgets import (
    QMainWindow, QToolBar, QDialog, QVBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QTextEdit, QCheckBox,
    QDialogButtonBox, QMessageBox,
)
try:
    from PySide6.QtGui import QAction
except ImportError:
    from PySide6.QtWidgets import QAction

from .app_state import GMSApplicationState

logger = logging.getLogger("gms.tp_ctrl")

_STAGES = [
    "Interpolation Stability", "Baseline Stability", "Smoothing Stability",
    "Fusion Evidence", "Cross Validation Evidence", "Reliability Evidence",
    "Signal Quality", "Spatial Stability", "Morphological Analysis",
    "Void Evidence", "Metallic Object Evidence", "Final Evidence Fusion",
]


class _TPWorkerSignals(QObject):
    stage_done = Signal(int, str)
    finished = Signal(list)      # list[TargetPossibility]
    failed = Signal(str)


class _TPWorker(QRunnable):
    """Runs the engine off the UI thread; emits per-stage progress."""
    def __init__(self, result_dict, gms_config):
        super().__init__()
        self.signals = _TPWorkerSignals()
        self._result = result_dict
        self._cfg = gms_config

    def run(self):
        try:
            from core.analysis.target_possibility import TargetPossibilityEngine
            engine = TargetPossibilityEngine(self._cfg)
            def _cb(i, name, done):
                if done:
                    self.signals.stage_done.emit(i, name)
            targets = engine.analyze(self._result, progress_cb=_cb)
            self.signals.finished.emit(targets)
        except Exception as e:
            logger.exception("[TP] analysis failed")
            self.signals.failed.emit(str(e))


class _ProgressDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Target Possibility — Analysis")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Running scientific evidence analysis…"))
        self._bar = QProgressBar(); self._bar.setRange(0, len(_STAGES))
        lay.addWidget(self._bar)
        self._list = QListWidget(); lay.addWidget(self._list)
        for name in _STAGES:
            it = QListWidgetItem(f"□  {name}")
            self._list.addItem(it)

    def mark(self, index: int, name: str):
        i = index - 1
        if 0 <= i < self._list.count():
            self._list.item(i).setText(f"✓  {name}")
        self._bar.setValue(min(index, len(_STAGES)))


class TargetPossibilityController(QObject):
    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w = window
        self._state = GMSApplicationState.instance()
        self._targets = []
        self._dialog = None
        self._install_button()

    # ── toolbar button ───────────────────────────────────────────────
    def _install_button(self):
        # Prefer an existing action if the .ui already defines one:
        act = self._w.findChild(QAction, "actionTargetPossibility")
        if act is None:
            tb = self._w.findChild(QToolBar)          # first toolbar
            act = QAction("Target Possibility", self._w)
            act.setObjectName("actionTargetPossibility")
            if tb is not None:
                tb.addAction(act)
        act.triggered.connect(self.run)
        self._action = act

    # ── run ─────────────────────────────────────────────────────
    def run(self):
        result = getattr(self._state, "last_result", None)
        if not result or result.get("baselined_grid") is None:
            QMessageBox.information(self._w, "Target Possibility",
                "Run a pipeline analysis first — no result to evaluate.")
            return
        self._dialog = _ProgressDialog(self._w)
        self._dialog.show()
        gms_config = result.get("gms_config", {})
        worker = _TPWorker(result, gms_config)
        worker.signals.stage_done.connect(self._dialog.mark)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_failed(self, msg: str):
        if self._dialog: self._dialog.close()
        QMessageBox.critical(self._w, "Target Possibility — Failed", msg)

    def _on_finished(self, targets: list):
        self._targets = targets
        if self._dialog: self._dialog.close()
        # publish to state so the heatmap overlay + any panel can read it
        self._state.__dict__["target_possibility"] = targets
        self._show_report(targets)
        self._push_overlay(targets)

    # ── report ──────────────────────────────────────────────────
    def _show_report(self, targets: list):
        dlg = QDialog(self._w)
        dlg.setWindowTitle("Target Possibility — Report")
        dlg.setMinimumSize(720, 560)
        lay = QVBoxLayout(dlg)

        filt = QCheckBox("Show only High / Very High possibility")
        lay.addWidget(filt)
        txt = QTextEdit(); txt.setReadOnly(True); lay.addWidget(txt)

        def _render():
            only_high = filt.isChecked()
            blocks = []
            for t in targets:
                if only_high and t.possibility_score < 75:
                    continue
                blocks.append(self._format_report_block(t))
            txt.setHtml("<hr>".join(blocks) if blocks else "<i>No targets in filter.</i>")
        filt.toggled.connect(_render)
        _render()

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    @staticmethod
    def _format_report_block(t) -> str:
        d = t.to_dict()
        ups = "".join(f"<li style='color:#2ECC71'>{r}</li>" for r in t.reasons_up) or "<li>—</li>"
        downs = "".join(f"<li style='color:#E74C3C'>{r}</li>" for r in t.reasons_down) or "<li>—</li>"
        warns = "".join(f"<li style='color:#E67E22'>{w}</li>" for w in t.warnings) or "<li>—</li>"
        depth = "—" if d["depth_m"] is None else f"{d['depth_m']} m ± {d['depth_uncertainty_m']}"
        return f"""
        <h3 style='color:{t.color}'>{t.target_id} — {t.classification}
            ({d['possibility_score']}/100)</h3>
        <b>Coordinates:</b> ({d['x_m']}, {d['y_m']}) m &nbsp;
        <b>Depth:</b> {depth} &nbsp; <b>Area:</b> {d['area_m2']} m² &nbsp;
        <b>Max amp:</b> {d['max_amplitude']}<br>
        <b>Confidence:</b> {d['confidence']:.2f} &nbsp;
        <b>Uncertainty:</b> {d['uncertainty']:.2f} &nbsp;
        <b>Fusion conf:</b> {d['fusion_confidence']:.2f} &nbsp;
        <b>Cross-val:</b> {d['cross_validation']}<br>
        <b>Interp stab:</b> {d['interpolation_stability']:.2f} &nbsp;
        <b>Baseline stab:</b> {d['baseline_stability']:.2f} &nbsp;
        <b>Smooth stab:</b> {d['smoothing_stability']:.2f} &nbsp;
        <b>Reliability:</b> {d['reliability_score']:.2f}<br>
        <b>Signal quality:</b> {d['signal_quality']:.2f} &nbsp;
        <b>Morphology:</b> {d['morphology_score']:.2f} &nbsp;
        <b>Void poss:</b> {d['void_possibility']:.2f} &nbsp;
        <b>Metallic poss:</b> {d['metallic_possibility']:.2f}<br>
        <b>Recommended action:</b> {t.recommended_action}
        <p><b>Increased score:</b><ul>{ups}</ul>
           <b>Decreased score:</b><ul>{downs}</ul>
           <b>Warnings:</b><ul>{warns}</ul></p>
        """

    # ── map overlay ──────────────────────────────────────────────
    def _push_overlay(self, targets: list):
        """
        Hand the colored targets to the HeatmapController if it exposes a hook.
        The heatmap draws a marker per target at (x_m,y_m) in target.color, and
        honors the hide-filters via a threshold. This stays display-only.
        """
        hm = None
        # controllers dict is stored on the window by bootstrap in many builds
        controllers = getattr(self._w, "_gms_controllers", None)
        if controllers:
            hm = controllers.get("heatmap")
        if hm is not None and hasattr(hm, "set_possibility_overlay"):
            hm.set_possibility_overlay(targets)
        else:
            logger.debug("[TP] heatmap overlay hook not present; report-only")