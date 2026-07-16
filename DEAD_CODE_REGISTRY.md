# GMS — Dead / Dormant Code Registry  v1.0

This document is the authoritative record of every module that exists in the
codebase but is **not reachable by any default user workflow**.  It is updated
whenever a module is connected, removed, or its status changes.

Statuses:
- 🟢 **LIVE**       — reached by at least one default workflow
- 🟡 **DORMANT**    — implemented, tested, but requires non-default activation
- 🔴 **DEAD**       — unreachable in current codebase; carries maintenance cost
- 🗑  **SCHEDULED**  — marked for removal in next major version

---

## Module Index

### `core/classifiers/topology.py` — 🟢 LIVE (was incorrectly flagged as dead)

| Field         | Value |
|---------------|-------|
| Class         | `TopologyValidator` |
| Used by       | `core/detectors/matched_cascade.py` (Stage 2.5 of cascade) |
| Activated via | Presets: `stable_v2`, `matched`, `matched_v2` |
| Status        | **LIVE** — imported and instantiated in `CascadedMatchedDetector.__init__()` |

**How to reach it:**
```python
from core.pipeline import build_pipeline
p = build_pipeline(config, preset="stable_v2")   # or "matched", "matched_v2"
bg, det = p.process_scan("myscan.csv")
# TopologyValidator runs as Stage 2.5 inside CascadedMatchedDetector
```

**Note:** The `stable` preset uses `detector="log"` which does NOT invoke
`TopologyValidator`.  The `stable_v2` preset specifically adds it.  A developer
deleting `topology.py` believing it is unused would break `stable_v2`,
`matched`, and `matched_v2` — all of which appear in `PRESETS` and can be
selected from the Pipeline Settings UI.

---

### `core/classifiers/orthogonal_fusion.py` — 🔴 DEAD

| Field         | Value |
|---------------|-------|
| Class         | `OrthogonalFusionEngine` |
| Imported by   | Nobody — zero imports across the codebase |
| Requires      | Two independent scan runs over the same area, scanned in perpendicular directions (X-pattern survey) |
| Why dead      | Single-CSV workflow never produces two orthogonal scans in one session; multi-scan UI (`tabScansCompare`) currently shows scans but does not route them through fuse() |

**Activation path (what must be wired):**
1. User completes two scans in perpendicular orientations
2. Both are loaded via `AdaptiveIngestionEngine` → pipeline → `BaselinedGrid`
3. `OrthogonalFusionEngine.fuse(result_x, result_y)` is called
4. Fused result replaces individual grids in `ScanCompareController`

**Recommended action:**  
Document in survey_ctrl as a Phase 3 feature.  Add an "Orthogonal survey" 
checkbox to `tabScanConfig` that triggers this path when ≥ 2 scans with 
perpendicular headings are loaded.  Do **not** delete — the algorithm is 
scientifically valuable for reducing false positives in high-noise fields.

---

### `core/io/` wrappers — 🔴 DEAD

| Field         | Value |
|---------------|-------|
| Modules       | `core/io/csv_reader.py`, `core/io/yaml_writer.py` (if present) |
| Imported by   | Nobody |
| Why dead      | `AdaptiveIngestionEngine` handles all ingestion internally using pandas directly; these wrappers were written for an older architecture |

**Recommended action:**  
Either delete, or make `AdaptiveIngestionEngine` delegate to them so they are
covered by `test_ingestion.py`.  Keeping dead wrappers increases maintenance
surface when pandas API changes.

---

### `core/command_history.py` — 🟡 DORMANT

| Field         | Value |
|---------------|-------|
| Classes       | `CommandHistory`, `UndoRedoController`, `ValidateTargetCommand` |
| Imported by   | `ui/integration/__init__.py` (bootstrapped), `inspector_ctrl.py` (used in `_on_confirm_dig`, `_on_reject`) |
| Why dormant   | `CommandHistory.execute()` is called but no undo/redo UI buttons are wired (no `btnUndo` / `btnRedo` found in widget scan) |

**Activation path:**
Add `btnUndo` and `btnRedo` QPushButtons to the toolbar in the `.ui` file.
Then in `bootstrap_integration`:
```python
undo_ctrl = controllers["undo_redo"]
# undo_ctrl already connects to CommandHistory.instance()
# just wire the buttons
```

---

### `core/ground_truth.py` validation panel — 🟢 LIVE (fixed in v3.6)

**Previously:** `GroundTruthWorkflow` was instantiated in `bootstrap_integration`
but `InspectorPanelController._wire_buttons()` did not include `btnValidate`,
and `set_ground_truth_workflow()` was never called.

**Fix applied (v3.6):**
- `inspector_ctrl._wire_buttons()` now includes `("btnValidate", self._on_validate)`
- `_on_validate()` method calls `self._gt_workflow.open_validation_panel(...)`
- `bootstrap_integration` calls `inspector_ctrl.set_ground_truth_workflow(gt_workflow)`
  immediately after the controllers dict is assembled

**Signal chain:**
```
btnValidate click
  → InspectorPanelController._on_validate()
  → GroundTruthWorkflow.open_validation_panel(anomaly_id, session_id, ...)
  → _ValidationDialog.exec()  [user fills form]
  → ValidationRecord.save()   [written to data/validations/]
  → CalibrationFeedback.suggest_adjustments()  [if ≥10 records]
```

---

### `ui/gms_controller.py` — Scan Registration — 🟢 LIVE (fixed in v3.6)

**Previously:** `ScanCompareController._build_ui()` had no Register button.
`core/registration.py` (`ScanRegistrationEngine`, `register_scan_pair`) was
fully implemented but unreachable from the UI.

**Fix applied (v3.6):**
- `_build_ui()` now creates a "REGISTRATION" panel with:
  - `btnRegisterScans` — triggers alignment
  - `chkShowDiff` — adds a Δ(B−A) difference overlay
  - `lblRegStatus` — shows method, quality score, and translation values
- `_on_register_pair()` calls `ScanRegistrationEngine.register(ref, mov)`,
  overwrites `mov.signal_grid` with the aligned grid, and emits a fault
  dialog if quality < 0.5

---

### `ui/integration/survey_ctrl.py` — Calibration apply path — 🟢 LIVE (fixed in v3.6)

**Previously:** `_on_apply_sensor()` and `_on_apply_soil()` stored calibration
into `state.__dict__` but never called `state.set_calibration()`, so
`calibration_state` remained `{}` and `calibration_changed` was never emitted.

**Fix applied (v3.6):**
Both methods now call `self._state.set_calibration(meta.calibration.to_dict())`
immediately after storing the calibration object, ensuring:
1. `calibration_state` always reflects the latest submitted values
2. `calibration_changed` signal fires so `StatusBarController` and
   `PipelineExecutionController` pick up the updated calibration

---

## Preset → Detector → Classifier Map

This table documents which preset activates which classifiers, resolving
the "is topology.py dead?" confusion permanently.

| Preset       | Interpolator      | Baseline         | Detector           | Classifiers active          |
|--------------|-------------------|------------------|--------------------|-----------------------------|
| `stable`     | cubic             | line_median      | log                | LoGDetector only            |
| `stable_v2`  | cubic             | line_median      | cascaded_matched   | DipoleValidator + **TopologyValidator** |
| `rbf`        | rbf_thin_plate    | none             | amplitude          | AmplitudeDetector only      |
| `sensitive`  | cubic             | line_median      | hybrid             | LoG + AmplitudeDetector     |
| `matched`    | cubic             | adaptive_local   | matched_dipole     | DipoleValidator + **TopologyValidator** |
| `matched_v2` | cubic             | multiscale       | cascaded_matched   | DipoleValidator + **TopologyValidator** |

`OrthogonalFusionEngine` is **not** in this table — no preset invokes it.

---

## Maintenance Rules

1. **Before deleting any module:** check this registry.  If status is 🟡 or
   🟢, search `matched_cascade.py` and `PRESETS` before concluding it is safe.
2. **When activating a 🔴 module:** update this file to 🟢 and document the
   exact signal chain.
3. **This file is updated in the same PR** as any wiring change.

---

*Last updated: v3.6 — fixes applied: calibration apply path, registration
UI, ground truth wiring, test suite addition.*
