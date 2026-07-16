"""
GMS — Event Bus  v1.0
======================
Central pub/sub event bus.  Any module can emit or subscribe without
direct coupling.  Built on top of Qt signals for thread-safety.

Events are typed strings for discoverability.
The bus routes them to the GMSApplicationState or direct subscribers.

Usage:
    bus = GMSEventBus.instance()
    bus.subscribe(GMS_EVENTS.SCAN_LOADED, my_handler)
    bus.emit_event(GMS_EVENTS.SCAN_LOADED, dataset=ds)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("gms.event_bus")


# ─────────────────────────────────────────────────────────────────────────────
# Event name constants
# ─────────────────────────────────────────────────────────────────────────────

class GMS_EVENTS:
    # Data lifecycle
    SCAN_LOADED              = "SCAN_LOADED"
    SCAN_CLEARED             = "SCAN_CLEARED"

    # Pipeline lifecycle
    PIPELINE_STARTED         = "PIPELINE_STARTED"
    PIPELINE_STAGE_STARTED   = "PIPELINE_STAGE_STARTED"
    PIPELINE_STAGE_PROGRESS  = "PIPELINE_STAGE_PROGRESS"
    PIPELINE_STAGE_COMPLETED = "PIPELINE_STAGE_COMPLETED"
    PIPELINE_STAGE_WARNING   = "PIPELINE_STAGE_WARNING"
    PIPELINE_STAGE_FAILED    = "PIPELINE_STAGE_FAILED"
    PIPELINE_FINISHED        = "PIPELINE_FINISHED"
    PIPELINE_CANCELLED       = "PIPELINE_CANCELLED"
    PIPELINE_FAILED          = "PIPELINE_FAILED"

    # Visualization
    VISUALIZATION_UPDATED    = "VISUALIZATION_UPDATED"
    RECOMPUTE_REQUESTED      = "RECOMPUTE_REQUESTED"
    LAYER_TOGGLED            = "LAYER_TOGGLED"

    # Anomaly / inspector
    ANOMALY_SELECTED         = "ANOMALY_SELECTED"
    ANOMALIES_UPDATED        = "ANOMALIES_UPDATED"
    CONFIDENCE_UPDATED       = "CONFIDENCE_UPDATED"

    # Compare
    SCAN_COMPARE_CHANGED     = "SCAN_COMPARE_CHANGED"

    # Calibration
    CALIBRATION_CHANGED      = "CALIBRATION_CHANGED"

    # Benchmark
    BENCHMARK_STARTED        = "BENCHMARK_STARTED"
    BENCHMARK_FINISHED       = "BENCHMARK_FINISHED"

    # Presets
    PRESET_CHANGED           = "PRESET_CHANGED"

    # Capabilities
    CAPABILITIES_CHANGED     = "CAPABILITIES_CHANGED"

    # Health / telemetry
    BACKEND_HEALTH_UPDATED   = "BACKEND_HEALTH_UPDATED"
    RELIABILITY_UPDATED      = "RELIABILITY_UPDATED"

    # Faults
    FAULT_RAISED             = "FAULT_RAISED"


# ─────────────────────────────────────────────────────────────────────────────
# GMSEventBus
# ─────────────────────────────────────────────────────────────────────────────

class _EventBusCore(QObject):
    """Qt object that holds the generic signal."""
    event_fired = Signal(str, object)   # event_name, payload dict


class GMSEventBus:
    """
    Lightweight pub/sub bus.  Thread-safe via Qt signal delivery.
    """
    _instance = None

    def __init__(self):
        self._core = _EventBusCore()
        self._subscribers: dict[str, list[Callable]] = {}
        self._core.event_fired.connect(self._dispatch)

    @classmethod
    def instance(cls) -> "GMSEventBus":
        if cls._instance is None:
            cls._instance = GMSEventBus()
        return cls._instance

    def subscribe(self, event_name: str, handler: Callable[[dict], None]):
        self._subscribers.setdefault(event_name, []).append(handler)

    def unsubscribe(self, event_name: str, handler: Callable):
        subs = self._subscribers.get(event_name, [])
        if handler in subs:
            subs.remove(handler)

    def emit_event(self, event_name: str, **payload):
        """Thread-safe emit — can be called from worker threads."""
        self._core.event_fired.emit(event_name, payload)

    def _dispatch(self, event_name: str, payload: Any):
        handlers = self._subscribers.get(event_name, [])
        for h in handlers:
            try:
                h(payload)
            except Exception as e:
                logger.error(f"[EventBus] Handler error for {event_name}: {e}")

        # Auto-bridge to GMSApplicationState
        self._bridge_to_state(event_name, payload)

    def _bridge_to_state(self, event_name: str, payload: dict):
        """
        Bridge bus events to GMSApplicationState setters so components that
        subscribe to state signals get consistent updates.
        """
        try:
            from .app_state import (
                GMSApplicationState, PipelineStatus,
                PipelineStageInfo, ReliabilityMetrics, BackendHealth
            )
            state = GMSApplicationState.instance()

            if event_name == GMS_EVENTS.SCAN_LOADED:
                state.set_dataset(payload.get("dataset"))

            elif event_name == GMS_EVENTS.SCAN_CLEARED:
                state.clear()

            elif event_name == GMS_EVENTS.PIPELINE_STARTED:
                state.set_pipeline_status(PipelineStatus.RUNNING)

            elif event_name == GMS_EVENTS.PIPELINE_STAGE_STARTED:
                info = PipelineStageInfo(
                    name=payload.get("stage", ""),
                    status="running",
                    progress=0.0,
                )
                state.update_stage(info)

            elif event_name == GMS_EVENTS.PIPELINE_STAGE_PROGRESS:
                info = PipelineStageInfo(
                    name=payload.get("stage", ""),
                    status="running",
                    progress=payload.get("progress", 0.0),
                )
                state.update_stage(info)
                # Overall progress = average of stages
                stages = state.pipeline_stages
                if stages:
                    avg = sum(s.progress for s in stages) / len(stages)
                    state.pipeline_progress.emit(avg)

            elif event_name == GMS_EVENTS.PIPELINE_STAGE_COMPLETED:
                info = PipelineStageInfo(
                    name=payload.get("stage", ""),
                    status="done",
                    progress=1.0,
                    duration_ms=payload.get("duration_ms", 0),
                    warnings=payload.get("warnings", []),
                )
                state.update_stage(info)

            elif event_name == GMS_EVENTS.PIPELINE_STAGE_FAILED:
                info = PipelineStageInfo(
                    name=payload.get("stage", ""),
                    status="failed",
                    progress=0.0,
                    error=payload.get("error", ""),
                )
                state.update_stage(info)

            elif event_name == GMS_EVENTS.PIPELINE_FINISHED:
                result = payload.get("result", {})
                state.set_pipeline_result(result)
                # Parse anomalies from result
                _build_anomaly_list(state, result)

            elif event_name == GMS_EVENTS.PIPELINE_FAILED:
                state.set_pipeline_status(PipelineStatus.FAILED)
                state.pipeline_failed.emit(payload.get("error", "Unknown error"))

            elif event_name == GMS_EVENTS.PIPELINE_CANCELLED:
                state.set_pipeline_status(PipelineStatus.CANCELLED)

            elif event_name == GMS_EVENTS.ANOMALY_SELECTED:
                state.set_selected_anomaly(payload.get("anomaly"))

            elif event_name == GMS_EVENTS.PRESET_CHANGED:
                state.set_preset(payload.get("preset", "stable"))

            elif event_name == GMS_EVENTS.CALIBRATION_CHANGED:
                state.set_calibration(payload.get("calibration", {}))

            elif event_name == GMS_EVENTS.RELIABILITY_UPDATED:
                raw = payload.get("reliability")
                if raw:
                    metrics = ReliabilityMetrics(
                        quality_label=getattr(raw, "quality_label", "UNKNOWN"),
                        is_reliable=getattr(raw, "is_reliable", True),
                        snr_mean=getattr(raw, "snr_mean", 0.0),
                        coverage=getattr(raw, "coverage", 0.0),
                        noise_floor=getattr(raw, "noise_floor", 0.0),
                        message=getattr(raw, "message", ""),
                    )
                    state.set_reliability(metrics)

            elif event_name == GMS_EVENTS.BACKEND_HEALTH_UPDATED:
                state.set_backend_health(payload.get("health", BackendHealth()))

            elif event_name == GMS_EVENTS.BENCHMARK_STARTED:
                state.benchmark_started.emit()

            elif event_name == GMS_EVENTS.BENCHMARK_FINISHED:
                state.benchmark_completed.emit(payload.get("results", {}))

            elif event_name == GMS_EVENTS.FAULT_RAISED:
                state.raise_fault(
                    payload.get("title", "Error"),
                    payload.get("message", ""),
                    payload.get("recovery", ""),
                )

        except Exception as e:
            logger.error(f"[EventBus] State bridge error for {event_name}: {e}")


def _build_anomaly_list(state, result: dict):
    """
    Convert pipeline result_dict confirmed_anomalies into AnomalyInfo objects.
    All values come from the real backend — no synthetic defaults beyond
    the stated fallback strings.

    Key: x, y are ALREADY in metres (converted in pipeline_exec._idx_to_metres).
    Key: explanation comes from ExplainabilityEngine (stored in a['explanation']).
    """
    from .app_state import AnomalyInfo
    anomalies = []
    for i, a in enumerate(result.get("confirmed_anomalies", [])):
        # Use ExplainabilityEngine output if present, else empty
        explanation = a.get("explanation", "")

        info = AnomalyInfo(
            anomaly_id=       a.get("anomaly_id",   f"T{i+1:03d}"),
            label=            a.get("label",         a.get("target_type", "UNKNOWN")),
            target_type=      a.get("target_type",   "UNKNOWN"),
            confidence=       float(a.get("confidence", a.get("combined_confidence", 0.0))),
            reliability=      float(a.get("reliability", 0.0)),
            snr=              float(a.get("snr",     a.get("mean_snr", 0.0))),
            x=                float(a.get("x",       0.0)),   # already metres
            y=                float(a.get("y",       0.0)),   # already metres
            depth_str=        a.get("depth_str",     "Calibration required"),
            dipole_score=     float(a.get("dipole_score",  0.0)),
            coherence=        float(a.get("coherence",     0.0)),
            fusion_boost=     float(a.get("fusion_boost",  0.0)),
            topology_status=  a.get("topology_status",     "unknown"),
            env_rejected=     bool(a.get("env_rejected",   False)),
            scan_confirmations=int(a.get("scan_confirmations", 1)),
            decision_reason=  explanation,
        )
        anomalies.append(info)

    state.set_anomaly_list(anomalies)
