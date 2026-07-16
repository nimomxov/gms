"""
GMS Integration Layer v3.6 (PATCHED)
FIX 7:  module-level `logger` defined — bootstrap_integration() previously
        called a bare `logger` that was never defined -> NameError on every launch.
FIX 10: removed redundant exec_c.result_ready -> set_pipeline_result wire.
        The bus (PIPELINE_FINISHED) is now the single delivery path, so the
        heatmap/3D/inspector rebuild once per run instead of twice.
"""

import logging

from .app_state import GMSApplicationState, PipelineStatus
from .event_bus import GMSEventBus, GMS_EVENTS
from .command_history import CommandHistory

logger = logging.getLogger("gms.bootstrap")   # FIX 7

__all__ = [
    "GMSApplicationState", "PipelineStatus",
    "GMSEventBus", "GMS_EVENTS",
    "CommandHistory",
    "bootstrap_integration",
]


def bootstrap_integration(window) -> dict:
    from .fault_manager import GMSFaultManager
    from .statusbar_ctrl import StatusBarController
    from .capability_gate import CapabilityGateController, AdaptiveImportWorkflow
    from .inspector_ctrl import InspectorPanelController
    from .heatmap_ctrl import HeatmapController
    from .benchmark_ctrl import BenchmarkController
    from .stage_progress_ctrl import PipelineStageProgressController
    from .pipeline_exec import PipelineExecutionController
    from .command_history import UndoRedoController
    from .ground_truth import GroundTruthWorkflow
    from .fusion_ctrl import FusionController
    from .survey_ctrl import SurveyController

    state = GMSApplicationState.instance()
    bus = GMSEventBus.instance()
    exec_c = PipelineExecutionController.instance()

    controllers = {
        "state": state,
        "bus": bus,
        "history": CommandHistory.instance(),
        "exec": exec_c,
        "fault_manager": GMSFaultManager(window),
        "statusbar": StatusBarController(window),
        "capability_gate": CapabilityGateController(window),
        "adaptive_import": AdaptiveImportWorkflow(window),
        "inspector": InspectorPanelController(window),
        "heatmap": HeatmapController(window),
        "benchmark": BenchmarkController(window),
        "stage_progress": PipelineStageProgressController(window),
        "undo_redo": UndoRedoController(window),
        "ground_truth": GroundTruthWorkflow(window),
        "fusion": FusionController(window),
        "survey": SurveyController(window),
    }

    # ── Wire GroundTruthWorkflow into InspectorPanelController ───────────
    inspector_ctrl = controllers.get("inspector")
    gt_workflow = controllers.get("ground_truth")
    if inspector_ctrl is not None and gt_workflow is not None:
        inspector_ctrl.set_ground_truth_workflow(gt_workflow)
        gt_workflow.validation_saved.connect(
            lambda rec: logger.info(
                f"[Bootstrap] Validation saved: {rec.anomaly_id} -> "
                f"{rec.actual_category.value}"
            )
        )
        logger.info("[Bootstrap] GroundTruthWorkflow wired to InspectorPanelController")

    # ── Volumetric engine + 3D keyboard nav ──────────────────────────────
    try:
        from core.volumetric import VolumetricEngine
        from PySide6.QtWidgets import QCheckBox, QPushButton, QSlider, QLabel, QComboBox

        vol = VolumetricEngine()
        attached = vol.attach(window) if hasattr(vol, "attach") else True
        if attached:
            controllers["volumetric"] = vol

            hm_ctrl = controllers.get("heatmap")
            if hm_ctrl is not None:
                def _vol_push(grid, anomalies, geometry, _vol=vol):
                    try:
                        _vol.set_scan(grid, anomalies, geometry=geometry)
                    except Exception as _e:
                        logger.debug(f"[3D sync] {_e}")
                hm_ctrl._vol_push_fn = _vol_push

            if hasattr(vol, "_view") and vol._view is not None:
                from .viewport_nav import attach_3d_nav
                nav_3d = attach_3d_nav(vol._view)
                if nav_3d:
                    controllers["nav_3d"] = nav_3d

            state.anomaly_selected.connect(
                lambda a: vol.select_anomaly(getattr(a, "anomaly_id", "") if a else ""))
            state.pipeline_completed.connect(lambda r: _push_3d(vol, r, state))

            cb_map = {
                "chk3dSignal": "signal", "chk3dBase": "baseline", "chk3dDig": "dig",
                "chk3dConf": "conf", "chk3dRaw": "raw", "chk3dGrid": "grid",
                "chk3dBlobs": "blobs",
            }
            for cb_name, layer in cb_map.items():
                cb = window.findChild(QCheckBox, cb_name)
                if cb:
                    cb.toggled.connect(lambda checked, ln=layer: vol.set_layer(ln, checked))

            for btn_name, preset in (
                ("btnResetCam", "reset"), ("btnTopView", "top"),
                ("btnSideView", "side"), ("btnPerspView", "perspective"),
            ):
                btn = window.findChild(QPushButton, btn_name)
                if btn:
                    btn.clicked.connect(lambda _, p=preset: vol.set_camera_preset(p))

            cmb_mode = window.findChild(QComboBox, "cmbRenderMode")
            if cmb_mode:
                if cmb_mode.count() == 0:
                    cmb_mode.addItems(["Surface", "Heightmap", "Volumetric", "Wireframe", "Points"])
                cmb_mode.currentTextChanged.connect(lambda txt: vol.set_render_mode(txt))

            sld = window.findChild(QSlider, "sldVertExag")
            if sld:
                def _vex(val):
                    vol.set_vertical_exag(val)
                    lbl = window.findChild(QLabel, "lblVexVal")
                    if lbl:
                        lbl.setText(f"{val}x")
                sld.valueChanged.connect(_vex)

    except Exception as e:
        logger.warning(f"[Bootstrap] Volumetric: {e}", exc_info=True)

    # ── Pipeline exec → state ────────────────────────────────────────────
    # FIX 10: do NOT also wire result_ready -> set_pipeline_result. The bus
    # PIPELINE_FINISHED bridge already calls set_pipeline_result AND builds
    # the anomaly list. Keeping both caused a double render per run.
    exec_c.failed.connect(state.pipeline_failed.emit)

    # ── Diagnostics ──────────────────────────────────────────────────────
    _wire_diagnostics(window)

    return controllers


def _push_3d(vol, result: dict, state):
    try:
        grid = result.get("baselined_grid")
        anomalies = state.anomaly_list
        geometry = state.__dict__.get("_geometry")
        if grid is not None:
            vol.set_scan(grid, anomalies, geometry=geometry)
    except Exception as e:
        logger.debug(f"[3D push] {e}")


def _wire_diagnostics(window):
    try:
        from PySide6.QtWidgets import QTextEdit
        from core.compute import roadmap_report, compute_info
        txt = window.findChild(QTextEdit, "textDiagnostics")
        if txt:
            info = compute_info()
            txt.setPlainText(
                f"Backend: {info.get('backend','?').upper()}\n"
                f"Device: {info.get('gpu_name', info.get('device','CPU'))}\n\n"
                + roadmap_report()
            )
    except Exception:
        pass
        
from .target_possibility_ctrl import TargetPossibilityController
controllers["target_possibility"] = TargetPossibilityController(window)
# expose controllers on the window so TP can reach the heatmap overlay hook:
window._gms_controllers = controllers
