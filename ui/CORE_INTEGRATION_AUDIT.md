# GMS Core Integration Audit — v3.5

## CONNECTED — modules actually executed in runtime path

| Module | Called from | Output consumed by |
|--------|------------|-------------------|
| `core/adaptive_ingestion.py` `AdaptiveIngestionEngine` | `AdaptiveImportWorkflow.run()` → `capability_gate.py` | `GMSApplicationState.set_dataset()` → `CapabilityGateController` |
| `core/schema/inspector.py` `CSVInspector` | inside `AdaptiveIngestionEngine.load()` | Schema detection fields → `SemanticFieldMapper` |
| `core/schema/mapper.py` `SemanticFieldMapper` | inside `AdaptiveIngestionEngine.load()` | `CapabilityExtractor` → `has_position`, `has_snr`, `has_heading` |
| `core/schema/capabilities.py` `CapabilityExtractor` | inside `AdaptiveIngestionEngine.load()` | `DeviceCapabilities` → `capability_gate.py` UI gating |
| `core/pipeline_composer.py` `DynamicPipelineComposer` | inside `AdaptiveIngestionEngine.load()` | `AdaptivePipeline.enabled_stages` → status bar, stage progress |
| `core/pipeline.py` `GMSPipeline.process_scan()` | `PipelineWorker.run()` → `pipeline_exec.py` | `BaselinedGrid` + `DetectionResult` → heatmap + 3D + inspector |
| `core/interpolators/plugins.py` | inside `GMSPipeline.process_scan()` | `GriddedScan.grid_x/y/z` → heatmap `extent`, 3D mesh |
| `core/baselines/` | inside `GMSPipeline.process_scan()` | `BaselinedGrid.grid_z` → heatmap render, 3D surface |
| `core/detectors/plugins.py` | inside `GMSPipeline.process_scan()` | `DetectionResult.anomalies` → cross-scan validator |
| `core/reliability.py` `ScanReliabilityEngine` | inside `GMSPipeline.process_scan()` | `reliability.quality_label`, `snr_mean` → inspector rel panel |
| `core/decision_engine.py` `CrossScanValidator` | `PipelineWorker.run()` after process_scan | `FinalReport` → result_dict → inspector, heatmap overlays |
| `core/explainability.py` `ExplainabilityEngine` | `PipelineWorker.run()` after CrossScanValidator | `explanation` field in result_dict → inspector explanation panel |
| `core/depth/inversion.py` `DepthInversionPlugin` | `PipelineWorker.run()` per anomaly | `depth_str` in result_dict → inspector `inspDepth` |
| `core/session.py` `ProcessingSession` | `PipelineWorker.run()` | JSON exported to `reports/`, exposed in diagnostics tab |
| `core/volumetric.py` `VolumetricEngine` | `bootstrap_integration()` → `__init__.py` | GLViewWidget in `vp3dPH` — surface + anomaly markers |
| `core/geometry.py` `GeometryReconstructor` | `btnSubmitGeometry` → controller | Synthetic XY applied to dataset before pipeline run |
| `core/compute.py` | `volumetric.py` surface render | Roadmap shown in diagnostics `textDiagnostics` |

---

## DISCONNECTED — dead modules never reached by current UI flow

| Module | Reason | Fix needed |
|--------|--------|-----------|
| `core/classifiers/topology.py` `TopologyValidator` | Called inside `detectors/matched_cascade.py` only — not in `log_detector` / `amplitude_detector` paths (used with `stable` preset) | Use `stable_v2` or `sensitive` preset which activates cascade detector |
| `core/classifiers/orthogonal_fusion.py` | Called only when ≥2 scans confirm overlapping anomalies | Requires multi-scan session — single CSV load doesn't trigger it |
| `core/registration.py` `ScanRegistrationEngine` | `tabScansCompare` compare controller not fully wired | Needs `ScanCompareController` to call `register_scan_pair()` before overlay |
| `core/async_engine/` | Not referenced in current integration layer | `PipelineWorker` (QRunnable) is used instead — covers the async requirement |
| `core/calibration/` | Calibration tab buttons wired but `CalibrationRegistry.save()` never called on apply | `_on_apply_sensor()` needs to call `SensorCalibration` with real spinbox values |
| `core/dataset/` | DatasetManager used in benchmark only | BenchmarkController wires correctly; main pipeline bypasses it |
| `core/io/` | CSV loading done via `ScanIngestionEngine` directly — `io/` wrappers unused | No action needed if ingestion works |
| `core/viz/` `CapabilityGatedVizEngine` | Fully bypassed — heatmap renders directly via matplotlib | No action needed; direct render is faster and avoids the line-chart fallback bug |

---

## BYPASSED — modules replaced by UI fallback logic (now fixed)

| Module | Old bypass | Fixed in v3.5 |
|--------|-----------|---------------|
| `core/explainability.py` | `event_bus._build_explanation()` wrote its own strings | Now: `ExplainabilityEngine.explain_anomaly()` called in `PipelineWorker`, result stored in `result_dict["explanation"]` |
| `core/depth/inversion.py` | `depth_str` hardcoded `"Calibration required"` always | Now: `DepthInversionPlugin.estimate_depth()` called per anomaly in worker |
| Grid-index → metres | `centroid_x/y` passed directly as "metres" (wrong) | Now: `_idx_to_metres(col, row, grid_x, grid_y)` converts correctly |
| `core/session.py` | Session created but never finalised or exported | Now: `session.finalize(result_dict)` + `session.export_json()` called in worker |
| `core/volumetric.py` | `VolumetricEngine(tab_3d_widget)` created duplicate canvas | Now: `vol.attach(window)` replaces `vp3dPH` in-place |

---

## BROKEN — modules failing at runtime (pre-v3.5)

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `CrossScanValidator()` no args | Constructor requires `config: dict` | Fixed: `CrossScanValidator(self._config)` |
| `FinalReport.to_dict()` missing | Method doesn't exist | Fixed: manual serialisation from `report.confirmed_anomalies` |
| Anomaly overlays at wrong position | `centroid_x/y` are grid indices, overlaid as metres | Fixed: `_idx_to_metres()` in `pipeline_exec.py` |
| `lbl.setPixmap(None)` crash | Qt API change | Fixed: `lbl.clear()` |
| `QAction` from `QtWidgets` | Moved to `QtGui` in PySide6 6.x | Fixed: `try QtGui, except QtWidgets` |
| Heatmap blank | `result.get("grid")` always `None` | Fixed: `baselined_grid` key carries real `BaselinedGrid` object |
| 3D duplicate viewport | `VolumetricEngine(container)` added GL widget alongside label | Fixed: `attach()` replaces label in-place |
| All viz controls disabled | `capability_gate._gate_all()` gated colormaps/sliders | Fixed: only `btnRunPipeline` is gated; all viz controls always enabled |

---

## Per-stage data flow (v3.5 runtime path)

```
cavity_survey.csv
  → AdaptiveIngestionEngine.load()
      → CSVInspector (schema detection)
      → SemanticFieldMapper (role → column)
      → CapabilityExtractor (grade, has_position, has_snr, …)
      → DynamicPipelineComposer (stages, disabled reasons)
      → AdaptiveScanDataset (x, y, values, capabilities, pipeline)
  → GMSApplicationState.set_dataset()
  → CapabilityGateController (gates confidence overlay only)

btnRunPipeline
  → PipelineWorker (QRunnable, off UI thread)
      → GMSPipeline.process_scan(filepath)
          → ScanIngestionEngine.load_csv()    [ingestion]
          → Preprocessing (DC remove, MAD filter)
          → Interpolator.interpolate()         [griddata_cubic / rbf]
          → BaselineRemover.remove()           [line_median / wavelet]
          → Detector.detect()                  [log / amplitude / cascade]
          → ScanReliabilityEngine.assess()     [SNR, Moran's I, coverage]
          → reliability penalty applied to anomaly confidence
          → returns (BaselinedGrid, DetectionResult)
      → CrossScanValidator(config).validate(results)
          → FinalReport (decision, confirmed_anomalies, single_detections)
      → _idx_to_metres() per anomaly            [grid index → metres]
      → ExplainabilityEngine.explain_anomaly()  [structured rationale]
      → DepthInversionPlugin.estimate_depth()   [depth_str per anomaly]
      → ProcessingSession.finalize() + export_json()
      → result_dict emitted via PIPELINE_FINISHED

PIPELINE_FINISHED
  → GMSEventBus._bridge_to_state()
      → state.set_pipeline_result(result_dict)
      → _build_anomaly_list() → AnomalyInfo list → state.set_anomaly_list()
  → HeatmapController._on_pipeline_completed()
      → grid.grid_z rendered with extent=[gx.min,gx.max,gy.min,gy.max]
      → anomaly crosshairs at real x_m, y_m
      → FigureCanvasQTAgg embedded in hmPlotLay
  → VolumetricEngine._push_3d()
      → surface mesh from grid_x/grid_y/grid_z (real metres)
      → anomaly GLScatterPlotItem at x_m, y_m
      → vp3dPH slot in vp3dWLay layout
  → InspectorPanelController._on_anomalies_updated()
      → listAnomalies populated
      → first anomaly auto-selected
      → all real metrics displayed (snr_robust, dipole_score, coherence, …)
      → ExplainabilityEngine text shown in inspExplanation
```
