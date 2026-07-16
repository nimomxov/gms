"""
GMS — Benchmark Controller  v1.0
==================================
Connects benchmark backend to the tabBenchmark UI tab.

Widget ObjectNames (from gms_main_window.ui):
  btnRunBench, benchProgress, cmbBenchDS, cmbBenchPipe,
  tableBenchResults, tprCLay, fprCLay, fnrCLay, accCLay,
  chkBenchMulti, chkParallel
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThreadPool
from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QProgressBar, QComboBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel,
)

from .app_state import GMSApplicationState
from .event_bus import GMSEventBus, GMS_EVENTS

logger = logging.getLogger("gms.benchmark_ctrl")


def _w(parent, cls, name):
    return parent.findChild(cls, name)


class BenchmarkController(QObject):
    """
    Drives the Benchmark tab — dataset selection, run, results display.
    Benchmark runs off the UI thread via BenchmarkWorker.
    """

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()
        self._bus   = GMSEventBus.instance()

        # Connect state signals
        self._state.benchmark_started.connect(self._on_started)
        self._state.benchmark_completed.connect(self._on_completed)

        # Wire buttons
        btn = _w(self._w, QPushButton, "btnRunBench")
        if btn:
            btn.clicked.connect(self._on_run)

        # Populate preset combo
        self._populate_combos()

        logger.info("[Benchmark] Controller attached")

    # ── Combo population ───────────────────────────────────────────────────

    def _populate_combos(self):
        try:
            from core.pipeline import PRESETS

            cmb = _w(self._w, QComboBox, "cmbBenchPipe")
            if cmb and cmb.count() == 0:
                for name in PRESETS:
                    cmb.addItem(name)

            cmb_ds = _w(self._w, QComboBox, "cmbBenchDS")
            if cmb_ds and cmb_ds.count() == 0:
                for label in ("Full suite (8 scenarios)", "Shallow targets only",
                              "Deep targets only", "Noise only"):
                    cmb_ds.addItem(label)
        except ImportError:
            pass

    # ── Run ────────────────────────────────────────────────────────────────

    def _on_run(self):
        cmb  = _w(self._w, QComboBox, "cmbBenchPipe")
        preset = cmb.currentText() if cmb else "stable"

        chk = _w(self._w, QWidget, "chkBenchMulti")
        n_scans = 3 if (chk and getattr(chk, "isChecked", lambda: False)()) else 1

        from .pipeline_exec import BenchmarkWorker
        worker = BenchmarkWorker(preset=preset, n_scans=n_scans)
        worker.signals.completed.connect(self._on_worker_completed)
        worker.signals.failed.connect(self._on_worker_failed)
        worker.signals.progress.connect(self._on_worker_progress)

        QThreadPool.globalInstance().start(worker)

        btn = _w(self._w, QPushButton, "btnRunBench")
        if btn:
            btn.setEnabled(False)
            btn.setText("Running…")

    # ── Worker callbacks ───────────────────────────────────────────────────

    def _on_worker_completed(self, results: dict):
        self._bus.emit_event(GMS_EVENTS.BENCHMARK_FINISHED, results=results)

    def _on_worker_failed(self, error: str):
        from .fault_manager import GMSFaultManager
        GMSFaultManager.raise_fault(
            "Benchmark Failed", error,
            "Check that synthetic dataset generation is available."
        )
        btn = _w(self._w, QPushButton, "btnRunBench")
        if btn:
            btn.setEnabled(True)
            btn.setText("Run Benchmark")

    def _on_worker_progress(self, done: int, total: int):
        prog = _w(self._w, QProgressBar, "benchProgress")
        if prog:
            prog.setRange(0, total)
            prog.setValue(done)

    # ── State handlers ─────────────────────────────────────────────────────

    def _on_started(self):
        prog = _w(self._w, QProgressBar, "benchProgress")
        if prog:
            prog.setRange(0, 0)   # indeterminate

    def _on_completed(self, results: dict):
        # Restore button
        btn = _w(self._w, QPushButton, "btnRunBench")
        if btn:
            btn.setEnabled(True)
            btn.setText("Run Benchmark")

        # Stop spinner
        prog = _w(self._w, QProgressBar, "benchProgress")
        if prog:
            prog.setRange(0, 100)
            prog.setValue(100)

        # Update metric cards
        self._update_metric_cards(results)

        # Update results table
        self._update_results_table(results)

    # ── UI update helpers ─────────────────────────────────────────────────

    def _update_metric_cards(self, results: dict):
        n_total = max(results.get("n_entries", 1), 1)
        n_by    = results.get("n_by_decision", {})
        n_dig   = n_by.get("DIG", 0)
        n_res   = n_by.get("RESCAN", 0)
        n_pass  = n_by.get("PASS", 0)

        tpr = n_dig / n_total
        fnr = n_res / n_total
        fpr = n_pass / n_total
        acc = (n_dig + n_pass) / n_total

        card_data = {
            "tprCLay": ("TPR", f"{tpr:.0%}"),
            "fprCLay": ("FPR", f"{fpr:.0%}"),
            "fnrCLay": ("FNR", f"{fnr:.0%}"),
            "accCLay": ("ACC", f"{acc:.0%}"),
        }
        for card_name, (_, value) in card_data.items():
            parent = _w(self._w, QWidget, card_name)
            if parent is None:
                continue
            for lbl in parent.findChildren(QLabel, "valueLabel"):
                lbl.setText(value)

        # Colour code: FPR red if > 5%
        fpr_card = _w(self._w, QWidget, "fprCLay")
        if fpr_card:
            color = "#E74C3C" if fpr > 0.05 else "#27AE60"
            for lbl in fpr_card.findChildren(QLabel, "valueLabel"):
                lbl.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _update_results_table(self, results: dict):
        tbl = _w(self._w, QTableWidget, "tableBenchResults")
        if tbl is None:
            return

        n_by = results.get("n_by_decision", {})
        rows = [
            ("Total entries",    str(results.get("n_entries", 0))),
            ("DIG decisions",    str(n_by.get("DIG", 0))),
            ("RESCAN decisions", str(n_by.get("RESCAN", 0))),
            ("PASS decisions",   str(n_by.get("PASS", 0))),
        ]

        tbl.setRowCount(len(rows))
        tbl.setColumnCount(2)
        tbl.setHorizontalHeaderLabels(["Metric", "Value"])

        for r, (metric, value) in enumerate(rows):
            tbl.setItem(r, 0, QTableWidgetItem(metric))
            tbl.setItem(r, 1, QTableWidgetItem(value))

        tbl.resizeColumnsToContents()
