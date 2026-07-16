# GMS Integration Layer — Architecture Guide  v3.1

## Overview

The integration layer transforms GMS from a loosely-coupled UI prototype into a
fully reactive scientific workstation. Every widget reflects real backend state.
No UI component ever directly calls backend code — all communication flows
through a central state model and event bus.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PySide6 UI Widgets                    │
│   (ObjectNames preserved — no structural changes)        │
└───────────────────────┬─────────────────────────────────┘
                        │ signals / slots only
┌───────────────────────▼─────────────────────────────────┐
│              Integration Controllers                      │
│                                                          │
│  HeatmapController        InspectorPanelController       │
│  CapabilityGateController BenchmarkController            │
│  StatusBarController      PipelineStageProgressCtrl      │
│  AdaptiveImportWorkflow   GMSFaultManager                │
└───────────────────────┬─────────────────────────────────┘
                        │ read/write
┌───────────────────────▼─────────────────────────────────┐
│              GMSApplicationState (singleton)             │
│                                                          │
│  current_dataset      pipeline_status    anomaly_list    │
│  visualization_state  reliability        backend_health  │
│  calibration_state    compare_mode       active_preset   │
│                                                          │
│  Signals: dataset_loaded, pipeline_status_changed,       │
│           anomaly_selected, visualization_changed, …     │
└───────────────────────┬─────────────────────────────────┘
                        │ emit_event / subscribe
┌───────────────────────▼─────────────────────────────────┐
│                   GMSEventBus                            │
│                                                          │
│  SCAN_LOADED  PIPELINE_STARTED  PIPELINE_STAGE_CHANGED   │
│  PIPELINE_FINISHED  ANOMALY_SELECTED  FAULT_RAISED …     │
└───────────────────────┬─────────────────────────────────┘
                        │ QThreadPool / QRunnable
┌───────────────────────▼─────────────────────────────────┐
│            PipelineExecutionController                   │
│                                                          │
│  PipelineWorker (QRunnable)                             │
│    → InstrumentedPipeline                               │
│      → GMSPipeline.process_scan()  (per stage)          │
│      → emits bus events at each stage                   │
│                                                          │
│  BenchmarkWorker (QRunnable)                            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   GMS Backend                            │
│                                                          │
│  AdaptiveIngestionEngine   GMSPipeline                  │
│  ScanReliabilityEngine     CrossScanValidator            │
│  CapabilityGatedVizEngine  DynamicPipelineComposer       │
│  (all existing backend — no modifications)               │
└─────────────────────────────────────────────────────────┘
```

---

## File Map

```
gms/ui/
  gms_controller.py                  ← original (preserved)
  gms_controller_integration_patch.py ← drop-in replacement
  integration/
    __init__.py                      ← bootstrap_integration()
    app_state.py                     ← GMSApplicationState singleton
    event_bus.py                     ← GMSEventBus + GMS_EVENTS
    pipeline_exec.py                 ← PipelineExecutionController + Workers
    inspector_ctrl.py                ← InspectorPanelController
    statusbar_ctrl.py                ← StatusBarController
    heatmap_ctrl.py                  ← HeatmapController (recompute graph)
    capability_gate.py               ← CapabilityGateController + AdaptiveImportWorkflow
    fault_manager.py                 ← GMSFaultManager
    benchmark_ctrl.py                ← BenchmarkController
    stage_progress_ctrl.py           ← PipelineStageProgressController
```

---

## How to Activate

In your application entry point, replace `GMSController` with
`IntegratedGMSController`:

```python
# Option A — use the patch module directly
from ui.gms_controller_integration_patch import create_integrated_app
app, window, ctrl = create_integrated_app(sys.argv)
window.show()
sys.exit(app.exec())

# Option B — bootstrap only (keep existing gms_controller.py)
from ui.gms_controller import create_app
from ui.integration import bootstrap_integration
app, window, ctrl = create_app(sys.argv)
controllers = bootstrap_integration(window)   # add this line
window.show()
sys.exit(app.exec())
```

---

## Incremental Recomputation Graph

The heatmap controller implements a tiered recompute model.
Only the minimum set of backend stages re-run when a control changes.

```
UI Control          Tier             Backend Effect
──────────────────────────────────────────────────────────
colormap            render_only      matplotlib re-render only
opacity             render_only      matplotlib re-render only
brightness/contrast render_only      matplotlib re-render only
contour toggle      render_only      matplotlib re-render only
layer checkboxes    render_only      matplotlib re-render only
smoothing slider    render_only      matplotlib re-render only

interp_method       interpolation    rerun: interp → baseline → detect
baseline method     baseline         rerun: baseline → detect
detector threshold  detector         rerun: detect only
sigma slider        detector         rerun: detect only
```

A 150 ms debounce prevents thrashing when sliders are dragged.

---

## Capability Gating Rules

```
Capability     Missing action
────────────────────────────────────────────────────────────
has_position   Disable tabHeatmap2D + tabExplorer3D
               Show warnNoXY label
               Line visualization only

has_snr        Disable confidence display (inspConfidence, chkLConfidence)
               Show warnNoSNR label

has_heading    Show warnNoHeading label
               Disable geometry reconstruction widgets

(none)         Disable btnRunPipeline until file is loaded
```

No fake values are ever shown. Disabled features always display an
explanatory label, not a zero or a placeholder.

---

## Async Processing

All heavy operations run on QThreadPool worker threads:

```
Heavy task              Worker class
────────────────────────────────────────────────────────────
Full pipeline           PipelineWorker (QRunnable)
Benchmark suite         BenchmarkWorker (QRunnable)
Incremental recompute   PipelineWorker (short run, same pool)
```

The UI thread only:
- handles Qt events
- updates widgets via signal/slot
- reads from GMSApplicationState

**The UI thread never calls any backend function directly.**

---

## Event Flow: CSV Load → Pipeline Run

```
1. User clicks btnOpenCSV / drags file
2. GMSController._on_open_csv()
3. AdaptiveImportWorkflow.run(filepath)
   → AdaptiveIngestionEngine.load()
   → GMSApplicationState.set_dataset()
   → GMSEventBus.emit(SCAN_LOADED)
4. CapabilityGateController._on_capabilities_changed()
   → gating rules applied to all widgets
5. User clicks btnRunPipeline
6. GMSController._on_run_pipeline()
7. PipelineExecutionController.run(scan_files, preset)
8. PipelineWorker runs on thread pool:
   PIPELINE_STARTED     → bus → state.pipeline_status = RUNNING
   STAGE_STARTED(x5)    → bus → stage table updated
   STAGE_COMPLETED(x5)  → bus → progress updated
   PIPELINE_FINISHED    → bus → state.set_pipeline_result(result)
9. Anomaly list extracted → InspectorPanelController populates
10. StatusBarController shows decision + confirmed count
```

---

## Fault Management

Any module anywhere can raise a fault:

```python
# From a backend module (no UI import needed):
from ui.integration.fault_manager import GMSFaultManager
GMSFaultManager.raise_fault(
    title="Interpolation Failed",
    message="Insufficient sample density for RBF.",
    recovery="Switch to Cubic interpolation in Pipeline Settings."
)

# Or via the event bus:
GMSEventBus.instance().emit_event(
    GMS_EVENTS.FAULT_RAISED,
    title="...", message="...", recovery="..."
)
```

The FaultManager presents a structured dialog with title, explanation,
and recovery action. It also installs a global sys.excepthook so no
exception ever silently crashes the application.

---

## Scientific Constraints (Preserved)

All original backend scientific rules remain unchanged:

- RESCAN is acceptable. False DIG is NOT acceptable.
- Confidence systems: reliability → explainability → reproducibility →
  FPR protection → THEN sensitivity.
- Missing telemetry fields → feature disabled, never estimated.
- Depth estimation stub remains honest: "Calibration required".
- Config hash preserved and displayed in status bar.

**viewport_nav.py — keyboard pan for 2D and 3D
MapKeyFilter is a QObject event filter installed on the canvas widget. It intercepts only Key_Left/Right/Up/Down — all mouse events, scroll, drag, context menus pass through untouched. Focus is set automatically on MouseButtonPress so arrow keys work immediately after clicking the map.
2D (HeatmapKeyNav): converts the pixel step to data units using the current xlim/ylim range, then calls ax.set_xlim / ax.set_ylim. Works correctly at any zoom level — the pan distance shrinks proportionally when zoomed in.
3D (GLViewKeyNav): calls GLViewWidget.pan(dx, dy, 0, relative=True). Pan distance scales with camera distance so it stays proportional to zoom. Falls back to the older API without the relative keyword if the pyqtgraph version doesn't support it.

**heatmap_ctrl.py — 2D rendering + nav wired in
After the FigureCanvasQTAgg is embedded on first render, attach_2d_nav(canvas, ax) is called once and the HeatmapKeyNav object is stored in self._nav_2d to prevent garbage collection. On subsequent renders (control changes), self._nav_2d.update_ax(self._ax) keeps the axes reference current.

**__init__.py — bootstrap with 3D nav + survey controller
After vol.attach(window) succeeds, attach_3d_nav(vol._view) is called and the GLViewKeyNav stored in controllers["nav_3d"]. SurveyController(window) is instantiated and stored in controllers["survey"].

**survey_ctrl.py — Phase 1/2 form controller
btSubmit → reads all geometry widgets → builds ScanGeometryConfig → validates → auto-fills spinFieldW/spinFieldL → stores in state._geometry and state._survey_metadata. Live update of field display fires on any spin box valueChanged. btnApplySensor / btnApplySoil submit calibration with full validation. validate_pipeline_ready() shows a blocking QMessageBox if either phase is incomplete.

**geometry.py — full data model
SurveyDirection enum with .from_combo_text() for comboBox_2. Field formula is fixed: width = (pts-1)×sample_dist, length = (lines-1)×line_spacing — radioV/radioH are scan_pattern metadata only. SurveyMetadata carries scan_id (12-char UUID) and ISO-8601 timestamp for future multi-scan fusion. Validation ranges: sensor_spacing 0.05–2.0m, mineralization 0–100, basalt 0–100, gain > 0, adc > 0.