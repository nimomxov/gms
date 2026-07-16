# GMS — Architecture Guide  v2.3
## Capability-Aware Geophysical Modeling System

**Production state: v2.3**  **Production preset: stable**

---

## System Philosophy

GMS is a universal geophysical visualization and analysis platform.
It adapts dynamically to whatever telemetry a device provides — from a
single signal channel to a full professional telemetry stream.

Core principle: If the telemetry does not provide a field, that feature
is disabled. The system NEVER estimates, fabricates, or hallucinates
missing data. Every disabled feature is explained to the user.

---

## Project Structure (v2.3)

gms/
  config/gms_config.yaml
  core/
    schema/                      [NEW] telemetry introspection layer
      inspector.py               CSVInspector — raw field discovery
      mapper.py                  SemanticFieldMapper — 50+ alias table
      capabilities.py            DeviceCapabilities + TelemetryFrame + CapabilityExtractor
    pipeline_composer.py         [NEW] DynamicPipelineComposer
    adaptive_ingestion.py        [NEW] AdaptiveIngestionEngine
    device_profiles.py           [NEW] DeviceProfileRegistry
    abstractions.py              ABCs for all pipeline stages
    pipeline.py                  GMSPipeline orchestrator + PRESETS
    ingestion.py                 legacy ScanIngestionEngine
    signal_processing.py         RBF interpolation + drift removal
    anomaly_detection.py         anomaly types
    decision_engine.py           CrossScanValidator + FinalReport
    reliability.py               ScanReliabilityEngine
    benchmark.py                 8-scenario synthetic benchmark
    interpolators/plugins.py
    baselines/plugins.py + adaptive.py + multiscale.py
    detectors/plugins.py + matched_filter.py + matched_cascade.py
    classifiers/topology.py + orthogonal_fusion.py
    depth/inversion.py           stub — scientifically honest
  devices/                       [NEW] device YAML profiles
    gms_ble.yaml
    okm_profile.yaml
    generic_csv.yaml
    uart_simple.yaml
  viz/
    visualization.py             legacy GeoVizEngine
    capability_viz.py            [NEW] CapabilityGatedVizEngine
    plugins/dig_marker.py + viz3d.py
  api/server.py
  demo_data/generate_demo.py
  main.py                        CLI v2.3

---

## Telemetry Grades

BASIC        signal only            → line visualization + amplitude detection
STANDARD     signal + x + y         → 2D heatmap + LoG detection
ADVANCED     + snr                  → confidence %, uncertainty radius
PROFESSIONAL + heading+baseline+stability+noise → full pipeline

---

## Data Flow

CSV/BLE/Serial
  -> CSVInspector         (raw field names + stats)
  -> DeviceProfileRegistry (profile match or auto-detect)
  -> SemanticFieldMapper   (alias resolution)
  -> CapabilityExtractor   (DeviceCapabilities + TelemetryGrade)
  -> DynamicPipelineComposer (enable/disable stages)
  -> TelemetryFrame[]      (unified internal model, all Optional)
  -> AdaptiveScanDataset   (frames + capabilities + pipeline + RawScan)
  -> GMSPipeline           (interpolate -> baseline -> detect -> decide)
  -> CapabilityGatedVizEngine
       IF has_position -> 2D heatmap + (optional) 3D explorer
       IF NOT          -> line chart with explanation
    **CRITICAL UPDATE [New] "tabScanConfig" is added contains Scan Geometry Parameters which is more accurate than x,y[ x=Number of Lines, y=Points per Line, Field Width (m), Field Length (m)] by clicking the "btSubmit" all Scan Geometry Parameters submited as scan coordination , so if the csv file has signal ONLY by filling those field the heatmap can be loaded very accuratly ,so NO NEED X,Y in csv files.
	

---

## Capability Gating

Missing: snr       -> confidence %, uncertainty radius DISABLED
[Old Method Scan Geometry Parameters is more accurate]Missing: x, y      -> 2D heatmap, 3D explorer DISABLED (line mode)
Missing: heading   -> path reconstruction, orientation DISABLED
Always  disabled   -> depth estimation (needs calibrated inversion)

Disabled features show an explanatory message, never fake values.
