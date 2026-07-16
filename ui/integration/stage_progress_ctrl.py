"""
GMS — Pipeline Stage Progress Controller  v1.0
================================================
Shows per-stage pipeline progress in the Diagnostics tab.
Drives:  tablePipeTimings, benchProgress (during pipeline run)

Widget ObjectNames (gms_main_window.ui):
  tablePipeTimings, benchProgress, pipeTimeLay
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QMainWindow, QTableWidget, QTableWidgetItem,
    QProgressBar, QWidget, QLabel,
)
from PySide6.QtGui import QColor

from .app_state import GMSApplicationState, PipelineStageInfo, PipelineStatus

logger = logging.getLogger("gms.stage_progress")

_STAGE_LABELS = {
    "pipeline_init":    "Initialising",
    "ingestion":        "File Ingestion",
    "preprocessing":    "Pre-processing",
    "interpolation":    "Interpolation",
    "baseline_removal": "Baseline Removal",
    "anomaly_detection":"Anomaly Detection",
    "reliability":      "Reliability Assessment",
    "cross_validation": "Cross-scan Validation",
}

_STATUS_COLORS = {
    "pending": QColor("#888888"),
    "running": QColor("#3498DB"),
    "done":    QColor("#27AE60"),
    "failed":  QColor("#E74C3C"),
    "skipped": QColor("#E67E22"),
}

def _w(parent, cls, name):
    return parent.findChild(cls, name)


class PipelineStageProgressController(QObject):

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()

        self._state.pipeline_stage_changed.connect(self._on_stage_changed)
        self._state.pipeline_status_changed.connect(self._on_status_changed)
        self._state.pipeline_progress.connect(self._on_overall_progress)

        self._init_table()
        logger.info("[StageProgress] Controller attached")

    def _init_table(self):
        tbl = _w(self._w, QTableWidget, "tablePipeTimings")
        if tbl is None:
            return
        tbl.setColumnCount(4)
        tbl.setHorizontalHeaderLabels(["Stage", "Status", "Time (ms)", "Warnings"])
        tbl.setRowCount(len(_STAGE_LABELS))
        for r, (key, label) in enumerate(_STAGE_LABELS.items()):
            tbl.setItem(r, 0, QTableWidgetItem(label))
            tbl.setItem(r, 1, QTableWidgetItem("—"))
            tbl.setItem(r, 2, QTableWidgetItem("—"))
            tbl.setItem(r, 3, QTableWidgetItem(""))
        tbl.resizeColumnsToContents()

    def _on_stage_changed(self, stage: PipelineStageInfo):
        tbl = _w(self._w, QTableWidget, "tablePipeTimings")
        if tbl is None:
            return

        label = _STAGE_LABELS.get(stage.name, stage.name)
        color = _STATUS_COLORS.get(stage.status, QColor("#888888"))

        # Find or add row
        row = None
        for r in range(tbl.rowCount()):
            item = tbl.item(r, 0)
            if item and item.text() == label:
                row = r
                break

        if row is None:
            row = tbl.rowCount()
            tbl.insertRow(row)
            tbl.setItem(row, 0, QTableWidgetItem(label))

        status_item = QTableWidgetItem(stage.status.upper())
        status_item.setForeground(color)
        tbl.setItem(row, 1, status_item)

        if stage.duration_ms > 0:
            tbl.setItem(row, 2, QTableWidgetItem(f"{stage.duration_ms} ms"))

        if stage.warnings:
            tbl.setItem(row, 3, QTableWidgetItem(", ".join(stage.warnings)))
        if stage.error:
            tbl.setItem(row, 3, QTableWidgetItem(f"ERROR: {stage.error}"))

    def _on_status_changed(self, status: PipelineStatus):
        if status == PipelineStatus.IDLE:
            self._init_table()

        prog = _w(self._w, QProgressBar, "benchProgress")
        if prog:
            if status == PipelineStatus.RUNNING:
                prog.setRange(0, 0)
                prog.setVisible(True)
            elif status in (PipelineStatus.COMPLETED, PipelineStatus.FAILED,
                            PipelineStatus.CANCELLED):
                prog.setRange(0, 100)
                prog.setValue(100 if status == PipelineStatus.COMPLETED else 0)
                prog.setVisible(False)

    def _on_overall_progress(self, progress: float):
        prog = _w(self._w, QProgressBar, "benchProgress")
        if prog and prog.maximum() == 0:
            # Switch from indeterminate to determinate
            prog.setRange(0, 100)
        if prog:
            prog.setValue(int(progress * 100))
