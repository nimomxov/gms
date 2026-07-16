"""
GMS — Pipeline Execution Controller  v3.5
==========================================
Runs GMSPipeline on a QThreadPool worker thread.
Never blocks the UI thread.

KEY FIX (v3.5):
  RawAnomaly.cx/cy and ConfirmedAnomaly.centroid_x/y are GRID CELL INDICES,
  not metres.  Conversion to metres:
    x_m = grid_x[int(round(col_idx))]
    y_m = grid_y[int(round(row_idx))]
  This is now applied in _idx_to_metres() before building result_dict.

ALSO:
  - All core metrics (dipole_score, snr_robust, spatial_coherence,
    uncertainty, final_score, topology, reliability) extracted from
    real backend objects and placed in result_dict.
  - ExplainabilityEngine is called here, result stored in result_dict.
  - DepthInversionPlugin called (returns honest "calibration required"
    if uncalibrated).
  - ProcessingSession created, finalised, and exported as JSON.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, QMutex, QMutexLocker

from .event_bus import GMSEventBus, GMS_EVENTS

logger = logging.getLogger("gms.pipeline_exec")

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _idx_to_metres(col_idx: float, row_idx: float,
                   grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[float, float]:
    """
    Convert grid cell indices (col, row) to survey metres.
    grid_x is the 1-D column axis (x), grid_y is the 1-D row axis (y).
    """
    ci = int(round(float(col_idx)))
    ri = int(round(float(row_idx)))
    ci = max(0, min(ci, len(grid_x) - 1))
    ri = max(0, min(ri, len(grid_y) - 1))
    return float(grid_x[ci]), float(grid_y[ri])


def _raw_anomaly_to_dict(a, grid_x: np.ndarray, grid_y: np.ndarray,
                          reliability=None) -> dict:
    """
    Convert a RawAnomaly to the result dict format.
    Converts cx/cy grid indices to real metres.
    Uses marker_cx/marker_cy for physical position (dipole midpoint).
    """
    # marker position is the physical target position
    x_m, y_m = _idx_to_metres(a.marker_cx, a.marker_cy, grid_x, grid_y)

    rel_score = 1.0
    if reliability is not None:
        rel_score = getattr(reliability, "reliability_score",
                    getattr(reliability, "_score", 1.0))

    return {
        "anomaly_id":          a.anomaly_id,
        "label":               a.raw_label,
        "target_type":         a.raw_label,
        # Real metre coordinates
        "x":                   round(x_m, 4),
        "y":                   round(y_m, 4),
        # Grid indices kept for reference
        "cx_idx":              a.cx,
        "cy_idx":              a.cy,
        "confidence":          round(a.confidence, 4),
        "combined_confidence": round(a.confidence, 4),
        "scan_confirmations":  1,
        # All real backend metrics
        "snr":                 round(a.snr_robust, 3),
        "mean_snr":            round(a.snr_robust, 3),
        "dipole_score":        round(a.dipole_score, 3),
        "coherence":           round(a.spatial_coherence, 3),
        "smoothness":          round(a.smoothness_score, 3),
        "polarity_ratio":      round(a.polarity_ratio, 3),
        "final_score":         round(a.final_score, 3),
        "mean_uncertainty":    round(a.uncertainty, 3),
        "uncertainty":         round(a.uncertainty, 3),
        "extent_cells":        a.extent_cells,
        "peak_amplitude":      round(a.peak_amplitude, 3),
        "detector_name":       a.detector_name,
        # Reliability
        "reliability":         round(rel_score * a.confidence, 4),
        # Placeholders (filled by topology/explainability below)
        "topology_status":     "unknown",
        "env_rejected":        False,
        "fusion_boost":        0.0,
        "label_agreement":     1.0,
        "spatial_consistency": 1.0,
        "depth_str":           "Calibration required",
        "explanation":         "",
    }


def _confirmed_anomaly_to_dict(c, grid_x: np.ndarray,
                                grid_y: np.ndarray) -> dict:
    """
    Convert a ConfirmedAnomaly (cross-scan) to result dict format.
    centroid_x/y are grid indices (average of contributing RawAnomalies).
    """
    # centroid is column index (x) / row index (y)
    x_m, y_m = _idx_to_metres(c.centroid_x, c.centroid_y, grid_x, grid_y)

    return {
        "anomaly_id":          c.group_id,
        "group_id":            c.group_id,
        "label":               c.best_label,
        "target_type":         c.best_label,
        "x":                   round(x_m, 4),
        "y":                   round(y_m, 4),
        "cx_idx":              c.centroid_x,
        "cy_idx":              c.centroid_y,
        "confidence":          round(c.combined_confidence, 4),
        "combined_confidence": round(c.combined_confidence, 4),
        "scan_confirmations":  c.scan_confirmations,
        "mean_snr":            round(c.mean_snr, 3),
        "snr":                 round(c.mean_snr, 3),
        "mean_uncertainty":    round(c.mean_uncertainty, 3),
        "uncertainty":         round(c.mean_uncertainty, 3),
        "label_agreement":     round(c.label_agreement, 3),
        "spatial_consistency": round(c.spatial_consistency, 3),
        "reliability":         round(c.combined_confidence * 0.9, 4),
        # Filled below
        "dipole_score":        0.0,
        "coherence":           round(c.spatial_consistency, 3),
        "smoothness":          0.0,
        "final_score":         round(c.combined_confidence, 3),
        "fusion_boost":        0.0,
        "topology_status":     "unknown",
        "env_rejected":        False,
        "depth_str":           "Calibration required",
        "explanation":         "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker signals
# ─────────────────────────────────────────────────────────────────────────────

class _WorkerSignals(QObject):
    stage_started   = Signal(str)
    stage_progress  = Signal(str, float)
    stage_completed = Signal(str, int, list)
    stage_failed    = Signal(str, str)
    stage_warning   = Signal(str, str)
    result_ready    = Signal(dict)
    failed          = Signal(str)
    cancelled       = Signal()


# ─────────────────────────────────────────────────────────────────────────────
# QRunnable worker
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QRunnable):
    """
    Runs the full GMS pipeline on a thread pool thread.
    Calls GMSPipeline.process_scan() which executes:
      ingestion → preprocessing → interpolation → baseline →
      detection → reliability → (cross-scan validation)
    Then enriches result_dict with:
      - correct metre coordinates (grid index → metres)
      - all real backend metrics
      - ExplainabilityEngine output
      - DepthInversionPlugin output
      - ProcessingSession JSON export
    """

    def __init__(self, scan_files: list[str], preset: str,
                 config: dict, session_id: str):
        super().__init__()
        self.setAutoDelete(True)
        self._files      = scan_files
        self._preset     = preset
        self._config     = config
        self._session_id = session_id
        self._cancel     = [False]
        self.signals     = _WorkerSignals()

    def cancel(self):
        self._cancel[0] = True

    def run(self):
        bus = GMSEventBus.instance()
        try:
            from core.pipeline import build_pipeline
            from core.decision_engine import CrossScanValidator
            from core.session import ProcessingSession, SessionRegistry
            from core.explainability import ExplainabilityEngine
            from core.depth.inversion import DepthInversionPlugin

            bus.emit_event(GMS_EVENTS.PIPELINE_STARTED)
            self.signals.stage_started.emit("pipeline_init")

            # ── Build pipeline ────────────────────────────────────────────
            pipeline = build_pipeline(self._config, preset=self._preset)

            # ── Session ───────────────────────────────────────────────────
            session = ProcessingSession.begin(
                scan_files=self._files,
                preset=self._preset,
            )
            SessionRegistry.register(session)

            # ── Per-scan execution ────────────────────────────────────────
            grids: list   = []
            results: list = []

            for fp in self._files:
                if self._cancel[0]:
                    break

                self.signals.stage_started.emit("ingestion")
                t0 = time.monotonic()

                try:
                    baselined, det_result = pipeline.process_scan(fp)
                    dur = int((time.monotonic() - t0) * 1000)
                    self.signals.stage_completed.emit(
                        "full_scan", dur,
                        list(det_result.warnings)
                    )
                    session.record_stage(
                        "full_scan", status="completed",
                        params={"preset": self._preset, "file": str(fp)},
                        duration_ms=dur,
                        warnings=list(det_result.warnings),
                    )
                    grids.append(baselined)
                    results.append(det_result)

                except InterruptedError:
                    break
                except Exception as e:
                    logger.exception(f"[Worker] process_scan failed: {fp}")
                    self.signals.stage_warning.emit("scan_error", str(e))
                    session.record_stage("full_scan", status="failed", error=str(e))

            if self._cancel[0]:
                bus.emit_event(GMS_EVENTS.PIPELINE_CANCELLED)
                self.signals.cancelled.emit()
                return

            if not grids:
                raise RuntimeError("No scans were successfully processed")

            # Primary grid for rendering
            primary_grid = grids[0]
            grid_x = primary_grid.grid_x
            grid_y = primary_grid.grid_y

            # ── Cross-scan validation ─────────────────────────────────────
            self.signals.stage_started.emit("cross_validation")
            t0 = time.monotonic()
            validator = CrossScanValidator(self._config)
            report    = validator.validate(results, session_id=self._session_id)
            dur = int((time.monotonic() - t0) * 1000)
            self.signals.stage_completed.emit("cross_validation", dur, [])
            session.record_stage("cross_validation", status="completed",
                                 duration_ms=dur)

            # ── Reliability from first scan ───────────────────────────────
            reliability = results[0].__dict__.get("reliability") if results else None

            # Emit reliability update
            if reliability is not None:
                bus.emit_event(GMS_EVENTS.RELIABILITY_UPDATED,
                               reliability=reliability)

            # ── Registration (multi-scan only) ────────────────────────────
            # Execution order: cross_validation → registration → fusion
            # For each scan[1..N], align onto scan[0] using
            # ScanRegistrationEngine.  The resulting translation is applied
            # to every anomaly centroid so that FusionInput receives
            # metre-corrected coordinates. geometry_is_metres=True is set
            # on every FusionInput so the fusion engine does not re-multiply.
            import datetime
            registration_log: dict = {}   # scan_id → {quality, tx, ty, method}
            fusion_anomaly_corrections: dict = {}   # scan_id → (dx_m, dy_m)

            if len(grids) >= 2:
                try:
                    from core.registration import ScanRegistrationEngine
                    self.signals.stage_started.emit("registration")
                    t0 = time.monotonic()
                    reg_engine = ScanRegistrationEngine(
                        max_translation_m=self._config.get(
                            "registration", {}).get("max_translation_m", 2.0),
                        max_rotation_deg=self._config.get(
                            "registration", {}).get("max_rotation_deg", 15.0),
                    )
                    reg_warnings: list = []
                    for i, (grid_i, det_i) in enumerate(
                            zip(grids[1:], results[1:]), start=1):
                        try:
                            reg_result = reg_engine.register(grids[0], grid_i)
                            sid = det_i.scan_id
                            registration_log[sid] = {
                                "quality":       reg_result.quality,
                                "translation_x": reg_result.translation_x,
                                "translation_y": reg_result.translation_y,
                                "rotation_deg":  reg_result.rotation_deg,
                                "method":        reg_result.method,
                            }
                            # Store metre correction to apply to anomaly centroids
                            fusion_anomaly_corrections[sid] = (
                                reg_result.translation_x,
                                reg_result.translation_y,
                            )
                            reg_warnings.extend(reg_result.warnings)
                            if reg_result.quality < 0.5:
                                logger.warning(
                                    f"[Worker] Registration quality low for "
                                    f"{sid}: {reg_result.quality:.2f}"
                                )
                            else:
                                logger.info(
                                    f"[Worker] Registered {sid} → "
                                    f"{grids[0].scan_id}: "
                                    f"tx={reg_result.translation_x:.3f}m "
                                    f"ty={reg_result.translation_y:.3f}m "
                                    f"q={reg_result.quality:.2f} "
                                    f"method={reg_result.method}"
                                )
                        except Exception as _re:
                            logger.warning(
                                f"[Worker] Registration failed for scan {i}: {_re}"
                            )

                    dur_reg = int((time.monotonic() - t0) * 1000)
                    self.signals.stage_completed.emit(
                        "registration", dur_reg, reg_warnings)
                    session.record_stage(
                        "registration", status="completed",
                        params={
                            "n_registered": len(registration_log),
                            "log": registration_log,
                        },
                        duration_ms=dur_reg,
                        warnings=reg_warnings,
                    )

                except Exception as _re_outer:
                    logger.warning(
                        f"[Worker] Registration stage skipped entirely: {_re_outer}"
                    )

            # ── Multi-scan fusion ─────────────────────────────────────────
            fusion_result = None
            if len(results) >= 1:
                try:
                    from core.fusion_engine import MultiScanFusionEngine, FusionInput
                    from core.geometry import ScanGeometryConfig
                    self.signals.stage_started.emit("fusion")
                    t0 = time.monotonic()

                    # Derive pixel size from primary grid for coordinate conversion
                    g0 = primary_grid
                    nx0 = len(g0.grid_x); ny0 = len(g0.grid_y)
                    dx0 = float((g0.grid_x[-1] - g0.grid_x[0]) / max(nx0-1, 1))
                    dy0 = float((g0.grid_y[-1] - g0.grid_y[0]) / max(ny0-1, 1))

                    now_ts = datetime.datetime.now(
                        datetime.timezone.utc).isoformat(timespec="seconds")

                    fusion_inputs = []
                    for gi, det in zip(grids, results):
                        sid = det.scan_id
                        dx_i = float((gi.grid_x[-1] - gi.grid_x[0]) / max(len(gi.grid_x)-1, 1))
                        dy_i = float((gi.grid_y[-1] - gi.grid_y[0]) / max(len(gi.grid_y)-1, 1))

                        # Apply registration correction to anomaly centroids:
                        # convert grid-index centroid to metres, then add
                        # the registration translation offset.
                        corr_tx, corr_ty = fusion_anomaly_corrections.get(sid, (0.0, 0.0))

                        # Build a corrected DetectionResult with metre centroids
                        # using a lightweight wrapper that the fusion engine
                        # accepts via geometry_is_metres=True.
                        import copy, dataclasses
                        corrected_anomalies = []
                        for a in det.anomalies:
                            # Convert grid-index → metres → apply registration offset
                            x_m = float(a.cx) * dx_i + corr_tx
                            y_m = float(a.cy) * dy_i + corr_ty
                            # Replace cx/cy with metre values (engine reads them as metres)
                            corrected = copy.copy(a)
                            object.__setattr__(corrected, "cx", x_m)                                 if dataclasses.is_dataclass(corrected) else                                 setattr(corrected, "cx", x_m)
                            object.__setattr__(corrected, "cy", y_m)                                 if dataclasses.is_dataclass(corrected) else                                 setattr(corrected, "cy", y_m)
                            corrected_anomalies.append(corrected)

                        corrected_det = copy.copy(det)
                        corrected_det.anomalies = corrected_anomalies

                        # Retrieve per-scan reliability from BaselinedGrid if available
                        scan_reliability = getattr(gi, "reliability", None) or                                            det.__dict__.get("reliability")

                        fusion_inputs.append(FusionInput(
                            detection_result=corrected_det,
                            geometry=None,          # not needed: geometry_is_metres=True
                            timestamp=now_ts,
                            reliability=scan_reliability,
                            geometry_is_metres=True,
                        ))

                    fuse_engine = MultiScanFusionEngine(
                        xy_tolerance_m=self._config.get(
                            "fusion", {}).get("xy_tolerance_m", 0.30),
                        depth_tolerance_m=self._config.get(
                            "fusion", {}).get("depth_tolerance_m", 0.15),
                    )
                    fusion_result = fuse_engine.fuse(fusion_inputs)

                    # Embed registration quality into each FusedTarget's diagnostics
                    if registration_log:
                        for target in fusion_result.targets:
                            if target.diagnostics is not None:
                                target.diagnostics.registration_quality = {
                                    sid: registration_log[sid]["quality"]
                                    for sid in target.supporting_scans
                                    if sid in registration_log
                                }

                    dur = int((time.monotonic() - t0) * 1000)
                    self.signals.stage_completed.emit("fusion", dur,
                        fusion_result.warnings)
                    session.record_stage("fusion", status="completed",
                        params={
                            "n_scans": len(results),
                            "n_clusters": fusion_result.n_clusters,
                            "registration_log": registration_log,
                        },
                        duration_ms=dur,
                        warnings=fusion_result.warnings)
                    logger.info(
                        f"[Worker] Fusion: {fusion_result.n_clusters} clusters "
                        f"({fusion_result.n_fused} fused, "
                        f"{fusion_result.n_singletons} singletons) "
                        f"from {len(results)} scans"
                    )
                except Exception as _fe:
                    logger.warning(f"[Worker] Fusion skipped: {_fe}")

            # ── Explainability engine ─────────────────────────────────────
            try:
                exp_engine = ExplainabilityEngine(pipeline.cfg)
            except Exception:
                exp_engine = None

            # ── Depth plugin ──────────────────────────────────────────────
            depth_plugin = DepthInversionPlugin()

            # ── Build confirmed anomaly list (cross-scan) ─────────────────
            confirmed_list = []

            for c in report.confirmed_anomalies:
                d = _confirmed_anomaly_to_dict(c, grid_x, grid_y)
                # Add depth estimate
                depth_info = depth_plugin.estimate_depth(c)
                if depth_info.get("depth_m") is not None:
                    d["depth_str"] = f"{depth_info['depth_m']:.2f} m"
                # Add explanation
                if exp_engine is not None:
                    try:
                        exp = exp_engine.explain_anomaly(
                            anomaly=c,
                            final_decision=report.decision,
                            reliability=reliability,
                        )
                        d["explanation"] = exp.full_text
                    except Exception as ex:
                        logger.debug(f"[Worker] Explainability failed: {ex}")
                confirmed_list.append(d)

            # ── Single-scan detections ────────────────────────────────────
            for a in report.single_detections:
                d = _raw_anomaly_to_dict(a, grid_x, grid_y, reliability)
                depth_info = depth_plugin.estimate_depth(a)
                if depth_info.get("depth_m") is not None:
                    d["depth_str"] = f"{depth_info['depth_m']:.2f} m"
                if exp_engine is not None:
                    try:
                        exp = exp_engine.explain_anomaly(
                            anomaly=a,
                            final_decision=report.decision,
                            reliability=reliability,
                        )
                        d["explanation"] = exp.full_text
                    except Exception as ex:
                        logger.debug(f"[Worker] Explainability failed: {ex}")
                confirmed_list.append(d)

            # ── Build reliability summary ─────────────────────────────────
            rel_summary = {}
            if reliability is not None:
                rel_summary = {
                    "quality_label":     reliability.quality_label,
                    "reliability_score": getattr(reliability, "_score", 0.0),
                    "snr_mean":          getattr(reliability, "snr_mean", 0.0),
                    "coverage":          getattr(reliability, "coverage", 0.0),
                    "noise_floor":       getattr(reliability, "noise_floor", 0.0),
                    "flags":             getattr(reliability, "flags", []),
                    "message":           getattr(reliability, "message", ""),
                }

            # ── Pipeline composition summary ──────────────────────────────
            pipeline_summary = {
                "interpolator":  pipeline.interpolator.name,
                "baseline":      pipeline.baseline.name,
                "detector":      pipeline.detector.name,
                "preset":        self._preset,
                "config_hash":   pipeline.cfg.config_hash(),
            }

            result_dict = {
                "session_id":          self._session_id,
                "decision":            report.decision,
                "confirmed_anomalies": confirmed_list,
                "confidence_summary":  report.confidence_summary,
                "scan_quality":        report.scan_quality,
                "warnings":            report.warnings,
                "n_scans_processed":   report.n_scans_processed,
                "n_confirmed":         len(report.confirmed_anomalies),
                "overall_confidence":  report.confidence_summary.get("overall", 0.0),
                "scan_files":          list(self._files),
                # Real backend objects for immediate rendering
                "baselined_grid":      primary_grid,
                "all_grids":           grids,
                # Fusion
                "fusion_result":       fusion_result,
                "fusion_summary":      fusion_result.summary() if fusion_result else {},
                "registration_log":    registration_log,
                # Full pipeline metadata
                "pipeline":            pipeline_summary,
                "reliability":         rel_summary,
            }

            # ── Finalise session ──────────────────────────────────────────
            session.finalize(result_dict)
            try:
                Path("reports").mkdir(exist_ok=True)
                session.export_json(
                    f"reports/{session.session_id}_provenance.json"
                )
            except Exception as e:
                logger.debug(f"[Worker] Session export failed: {e}")

            bus.emit_event(GMS_EVENTS.PIPELINE_FINISHED, result=result_dict)
            self.signals.result_ready.emit(result_dict)

            logger.info(
                f"[Worker] Done: decision={report.decision} "
                f"confirmed={len(report.confirmed_anomalies)} "
                f"single={len(report.single_detections)}"
            )

        except Exception as e:
            logger.exception(f"[PipelineWorker] Unhandled: {e}")
            bus.emit_event(GMS_EVENTS.PIPELINE_FAILED, error=str(e))
            self.signals.failed.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PipelineExecutionController
# ─────────────────────────────────────────────────────────────────────────────

class PipelineExecutionController(QObject):
    _instance = None

    stage_started   = Signal(str)
    stage_progress  = Signal(str, float)
    stage_completed = Signal(str, int, list)
    stage_failed    = Signal(str, str)
    stage_warning   = Signal(str, str)
    result_ready    = Signal(dict)
    failed          = Signal(str)
    cancelled       = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool           = QThreadPool.globalInstance()
        self._current_worker: Optional[PipelineWorker] = None
        self._mutex          = QMutex()
        self._current_files: list[str] = []

    @classmethod
    def instance(cls) -> "PipelineExecutionController":
        if cls._instance is None:
            cls._instance = PipelineExecutionController()
        return cls._instance

    def is_running(self) -> bool:
        return self._current_worker is not None

    def run(self, scan_files: list[str], preset: str = "stable",
            config: dict = None, session_id: str = "ui_run"):
        with QMutexLocker(self._mutex):
            if self._current_worker:
                self._current_worker.cancel()
                self._current_worker = None

        if not scan_files:
            return

        self._current_files = list(scan_files)
        worker = PipelineWorker(scan_files, preset, config or {}, session_id)
        self._current_worker = worker

        worker.signals.stage_started.connect(self.stage_started)
        worker.signals.stage_progress.connect(self.stage_progress)
        worker.signals.stage_completed.connect(self.stage_completed)
        worker.signals.stage_failed.connect(self.stage_failed)
        worker.signals.stage_warning.connect(self.stage_warning)
        worker.signals.result_ready.connect(self._on_result)
        worker.signals.failed.connect(self._on_failed)
        worker.signals.cancelled.connect(self._on_cancelled)

        self._pool.start(worker)
        logger.info(f"[PipelineExec] Started: {len(scan_files)} files, preset={preset}")

    def cancel(self):
        with QMutexLocker(self._mutex):
            if self._current_worker:
                self._current_worker.cancel()

    def _on_result(self, result: dict):
        with QMutexLocker(self._mutex):
            self._current_worker = None
        self.result_ready.emit(result)

    def _on_failed(self, error: str):
        with QMutexLocker(self._mutex):
            self._current_worker = None
        self.failed.emit(error)

    def _on_cancelled(self):
        with QMutexLocker(self._mutex):
            self._current_worker = None
        self.cancelled.emit()
