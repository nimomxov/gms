"""
GMS — Application State Engine  v1.0
=====================================
Single authoritative state model for the entire application.

ALL frontend components read from this object.
ALL backend results write THROUGH this object via signals.
NO direct widget ↔ backend coupling allowed.

Usage:
    state = GMSApplicationState.instance()
    state.pipeline_status_changed.connect(my_slot)
    state.set_pipeline_status(PipelineStatus.RUNNING)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("gms.app_state")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStatus(Enum):
    IDLE      = auto()
    LOADING   = auto()
    RUNNING   = auto()
    COMPLETED = auto()
    FAILED    = auto()
    CANCELLED = auto()


class TaskState(Enum):
    QUEUED    = auto()
    RUNNING   = auto()
    CANCELLED = auto()
    FAILED    = auto()
    COMPLETED = auto()


class TelemetryGradeEnum(Enum):
    NONE         = 0
    BASIC        = 1
    STANDARD     = 2
    ADVANCED     = 3
    PROFESSIONAL = 4


# ─────────────────────────────────────────────────────────────────────────────
# State data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisualizationState:
    colormap: str          = "plasma"
    opacity: float         = 1.0
    show_contours: bool    = False
    show_grid: bool        = True
    show_anomalies: bool   = True
    show_baseline: bool    = False
    show_confidence: bool  = True
    show_dig_zones: bool   = True
    show_raw_pts: bool     = False
    interp_method: str     = "cubic"
    brightness: float      = 0.5
    contrast: float        = 0.5
    smoothing: float       = 0.0
    vert_exag: int         = 3


@dataclass
class PipelineStageInfo:
    name: str
    status: str       = "pending"   # pending | running | done | failed | skipped
    progress: float   = 0.0         # 0–1
    duration_ms: int  = 0
    warnings: list    = field(default_factory=list)
    error: str        = ""


@dataclass
class AnomalyInfo:
    anomaly_id: str
    label: str
    target_type: str
    confidence: float
    reliability: float
    snr: float
    x: float
    y: float
    depth_str: str
    dipole_score: float
    coherence: float
    fusion_boost: float
    topology_status: str
    env_rejected: bool
    scan_confirmations: int
    decision_reason: str


@dataclass
class CompareModeState:
    active: bool                 = False
    scan_ids: list               = field(default_factory=list)
    show_difference: bool        = False
    synchronized_cursors: bool   = True


@dataclass
class ReliabilityMetrics:
    quality_label: str = "UNKNOWN"
    is_reliable: bool  = True
    snr_mean: float    = 0.0
    coverage: float    = 0.0
    noise_floor: float = 0.0
    message: str       = ""


@dataclass
class BackendHealth:
    interpolator_ok: bool = True
    baseline_ok: bool     = True
    detector_ok: bool     = True
    memory_ok: bool       = True
    cpu_pct: float        = 0.0
    ram_pct: float        = 0.0
    fps: float            = 0.0
    queue_depth: int      = 0


# ─────────────────────────────────────────────────────────────────────────────
# GMSApplicationState — singleton
# ─────────────────────────────────────────────────────────────────────────────

class GMSApplicationState(QObject):
    """
    Single authoritative state model.  Every UI component subscribes to signals
    emitted here.  No component mutates state directly — they call setters.
    """

    _instance: Optional["GMSApplicationState"] = None

    # ── Signals ───────────────────────────────────────────────────────────────

    # Dataset
    dataset_loaded          = Signal(object)   # AdaptiveScanDataset
    dataset_cleared         = Signal()

    # Pipeline lifecycle
    pipeline_status_changed = Signal(object)   # PipelineStatus
    pipeline_stage_changed  = Signal(object)   # PipelineStageInfo
    pipeline_progress       = Signal(float)    # 0.0–1.0
    pipeline_completed      = Signal(dict)     # full result dict
    pipeline_failed         = Signal(str)      # error message

    # Visualization
    visualization_changed   = Signal(object)   # VisualizationState
    recompute_requested     = Signal(str)      # stage name to recompute from

    # Anomalies
    anomaly_selected        = Signal(object)   # AnomalyInfo | None
    anomalies_updated       = Signal(list)     # list[AnomalyInfo]

    # Calibration
    calibration_changed     = Signal(dict)

    # Benchmark
    benchmark_started       = Signal()
    benchmark_completed     = Signal(dict)

    # Compare mode
    compare_state_changed   = Signal(object)   # CompareModeState

    # Multi-scan fusion
    fusion_changed          = Signal(object)   # FusionResult

    # Reliability
    reliability_updated     = Signal(object)   # ReliabilityMetrics

    # Backend health / telemetry
    backend_health_updated  = Signal(object)   # BackendHealth

    # Preset
    preset_changed          = Signal(str)      # preset name

    # Capability gating
    capabilities_changed    = Signal(object)   # DeviceCapabilities

    # Confidence
    confidence_updated      = Signal(float, str)  # value, explanation

    # Fault
    fault_raised            = Signal(str, str, str)  # title, message, recovery

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)

        # Core state
        self.current_dataset      = None        # AdaptiveScanDataset
        self.current_pipeline     = None        # GMSPipeline
        self.active_preset        = "stable"
        self.active_scan          = None
        self.selected_anomaly     = None        # AnomalyInfo
        self.pipeline_status      = PipelineStatus.IDLE
        self.visualization_state  = VisualizationState()
        self.active_layers        = {}
        self.calibration_state    = {}
        self.telemetry_grade      = TelemetryGradeEnum.NONE
        self.backend_health       = BackendHealth()
        self.reliability_metrics  = ReliabilityMetrics()
        self.compare_mode_state   = CompareModeState()
        self.fusion_result        = None         # FusionResult | None
        self.pipeline_stages      = []          # list[PipelineStageInfo]
        self.last_result          = None        # last pipeline result dict
        self.anomaly_list         = []          # list[AnomalyInfo]

        # Incremental recompute graph
        # Defines what must be recomputed when a control changes
        self._recompute_graph = {
            "colormap":      None,          # render only — no backend recompute
            "opacity":       None,
            "contours":      None,
            "smoothing":     None,
            "brightness":    None,
            "contrast":      None,
            "interp_method": "interpolation",
            "baseline":      "baseline",
            "threshold":     "detector",
            "sigma":         "detector",
        }

    @classmethod
    def instance(cls) -> "GMSApplicationState":
        if cls._instance is None:
            cls._instance = GMSApplicationState()
        return cls._instance

    # ── Setters (emit signals on change) ──────────────────────────────────────

    def set_dataset(self, dataset):
        self.current_dataset = dataset
        if dataset is not None:
            from core.schema.capabilities import TelemetryGrade
            grade_map = {
                TelemetryGrade.BASIC:        TelemetryGradeEnum.BASIC,
                TelemetryGrade.STANDARD:     TelemetryGradeEnum.STANDARD,
                TelemetryGrade.ADVANCED:     TelemetryGradeEnum.ADVANCED,
                TelemetryGrade.PROFESSIONAL: TelemetryGradeEnum.PROFESSIONAL,
            }
            self.telemetry_grade = grade_map.get(
                dataset.capabilities.grade, TelemetryGradeEnum.BASIC
            )
            self.capabilities_changed.emit(dataset.capabilities)
            self.dataset_loaded.emit(dataset)
        else:
            self.telemetry_grade = TelemetryGradeEnum.NONE
            self.dataset_cleared.emit()
        logger.debug(f"[State] dataset set: {dataset and dataset.scan_id}")

    def set_pipeline_status(self, status: PipelineStatus):
        self.pipeline_status = status
        self.pipeline_status_changed.emit(status)
        logger.debug(f"[State] pipeline_status → {status.name}")

    def update_stage(self, stage: PipelineStageInfo):
        # Upsert in stage list
        for i, s in enumerate(self.pipeline_stages):
            if s.name == stage.name:
                self.pipeline_stages[i] = stage
                break
        else:
            self.pipeline_stages.append(stage)
        self.pipeline_stage_changed.emit(stage)

    def set_pipeline_result(self, result: dict):
        self.last_result = result
        self.pipeline_status = PipelineStatus.COMPLETED
        self.pipeline_completed.emit(result)

    def set_visualization(self, key: str, value):
        """Update one visualization parameter and emit appropriate signal."""
        setattr(self.visualization_state, key, value)
        self.visualization_changed.emit(self.visualization_state)

        recompute_stage = self._recompute_graph.get(key)
        if recompute_stage:
            self.recompute_requested.emit(recompute_stage)

    def set_selected_anomaly(self, anomaly: Optional[AnomalyInfo]):
        self.selected_anomaly = anomaly
        self.anomaly_selected.emit(anomaly)

    def set_anomaly_list(self, anomalies: list):
        self.anomaly_list = anomalies
        self.anomalies_updated.emit(anomalies)

    def set_reliability(self, metrics: ReliabilityMetrics):
        self.reliability_metrics = metrics
        self.reliability_updated.emit(metrics)

    def set_backend_health(self, health: BackendHealth):
        self.backend_health = health
        self.backend_health_updated.emit(health)

    def set_preset(self, preset_name: str):
        self.active_preset = preset_name
        self.preset_changed.emit(preset_name)

    def set_calibration(self, cal_dict: dict):
        self.calibration_state = cal_dict
        self.calibration_changed.emit(cal_dict)

    def set_compare_mode(self, state: CompareModeState):
        self.compare_mode_state = state
        self.compare_state_changed.emit(state)

    def set_fusion_result(self, result) -> None:
        """Store a FusionResult and notify all subscribers."""
        self.fusion_result = result
        self.fusion_changed.emit(result)

    def raise_fault(self, title: str, message: str, recovery: str = ""):
        logger.error(f"[Fault] {title}: {message}")
        self.fault_raised.emit(title, message, recovery)

    def clear(self):
        self.current_dataset = None
        self.last_result = None
        self.anomaly_list = []
        self.selected_anomaly = None
        self.pipeline_stages = []
        self.pipeline_status = PipelineStatus.IDLE
        self.dataset_cleared.emit()
        self.pipeline_status_changed.emit(PipelineStatus.IDLE)
