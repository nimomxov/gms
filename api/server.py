"""
GMS API Layer — FastAPI Server
Endpoints: upload_scans, analyze, results, heatmap, report
Decoupled from core engine for Android/Windows/Web compatibility.
"""

import uuid
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Core engine imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ingestion import ScanIngestionEngine, DataIngestionError
from core.signal_processing import SignalProcessor
from core.anomaly_detection import AnomalyDetector
from core.decision_engine import CrossScanValidator
from viz.visualization import GeoVizEngine

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/gms_api.log", mode="a"),
    ]
)
logger = logging.getLogger("gms.api")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config" / "gms_config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# ── In-memory session store (replace with DB in production) ──────────────────
sessions: dict[str, dict] = {}

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GMS — Geophysical Modeling System API",
    description="Research-grade multi-scan geophysical signal analyzer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Engine instances ──────────────────────────────────────────────────────────
ingestion_engine = ScanIngestionEngine(CONFIG)
signal_processor = SignalProcessor(CONFIG)
anomaly_detector = AnomalyDetector(CONFIG)
cross_validator = CrossScanValidator(CONFIG)
viz_engine = GeoVizEngine(CONFIG, output_dir="reports")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "system": "GMS - Geophysical Modeling System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/upload_scans", "/analyze", "/results/{id}",
                      "/heatmap/{id}", "/report/{id}"]
    }


@app.post("/upload_scans", tags=["Data"])
async def upload_scans(files: list[UploadFile] = File(...)):
    """
    Upload one or more scan CSV files.
    Returns a session_id to use in subsequent calls.
    """
    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 scan files per session")

    session_id = str(uuid.uuid4())[:12]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gms_{session_id}_"))

    saved_files = []
    errors = []

    for uf in files:
        if not uf.filename.lower().endswith(".csv"):
            errors.append(f"{uf.filename}: not a CSV file")
            continue
        dest = tmp_dir / uf.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(uf.file, f)
        saved_files.append(str(dest))

    if not saved_files:
        raise HTTPException(400, f"No valid CSV files uploaded. Errors: {errors}")

    sessions[session_id] = {
        "session_id": session_id,
        "status": "uploaded",
        "uploaded_at": datetime.utcnow().isoformat(),
        "files": saved_files,
        "tmp_dir": str(tmp_dir),
        "errors": errors,
        "result": None,
        "heatmap_paths": {},
    }

    logger.info(f"Session {session_id}: {len(saved_files)} files uploaded")

    return {
        "session_id": session_id,
        "n_files_accepted": len(saved_files),
        "files": [Path(f).name for f in saved_files],
        "warnings": errors,
        "next_step": f"POST /analyze with session_id={session_id}",
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze(session_id: str, background_tasks: BackgroundTasks):
    """
    Run the full GMS analysis pipeline on an uploaded session.
    """
    if session_id not in sessions:
        raise HTTPException(404, f"Session not found: {session_id}")

    session = sessions[session_id]
    if session["status"] == "processing":
        return {"message": "Already processing", "session_id": session_id}

    session["status"] = "processing"

    # Run pipeline (synchronous for simplicity — wrap in thread for production)
    try:
        result = _run_pipeline(session_id, session["files"])
        session["result"] = result
        session["status"] = "complete"
        logger.info(f"Session {session_id}: analysis complete. Decision={result['decision']}")
        return {
            "session_id": session_id,
            "status": "complete",
            "decision": result["decision"],
            "summary": result["confidence_summary"],
        }
    except Exception as e:
        session["status"] = "error"
        session["error"] = str(e)
        logger.error(f"Session {session_id} failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {e}")


@app.get("/results/{session_id}", tags=["Results"])
def get_results(session_id: str):
    """
    Retrieve full structured analysis results.
    """
    session = _get_complete_session(session_id)
    return JSONResponse(content=session["result"])


@app.get("/heatmap/{session_id}", tags=["Visualization"])
def get_heatmap(session_id: str, scan_index: int = 0):
    """
    Return the heatmap PNG for a specific scan in the session.
    """
    session = _get_complete_session(session_id)
    paths = session.get("heatmap_paths", {})

    if not paths:
        raise HTTPException(404, "No heatmap generated yet")

    keys = list(paths.keys())
    if scan_index >= len(keys):
        raise HTTPException(404, f"Scan index {scan_index} out of range ({len(keys)} scans)")

    scan_id = keys[scan_index]
    png = paths[scan_id].get("png")
    if not png or not Path(png).exists():
        raise HTTPException(404, "Heatmap file not found on disk")

    return FileResponse(png, media_type="image/png",
                        filename=f"gms_{scan_id}_heatmap.png")


@app.get("/report/{session_id}", tags=["Results"])
def get_report(session_id: str):
    """
    Return a human-readable scientific report as JSON.
    Includes decision rationale, anomaly details, scan quality.
    """
    session = _get_complete_session(session_id)
    result = session["result"]

    # Build a clean, annotated report
    report = {
        "report_title": "GMS Geophysical Analysis Report",
        "system_version": "1.0.0",
        "session_id": session_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scientific_disclaimer": (
            "This report is based on statistical signal analysis only. "
            "No physical depth estimation is provided without explicit sensor calibration. "
            "All decisions should be validated by a qualified geophysicist."
        ),
        "decision": result["decision"],
        "decision_rationale": _decision_rationale(result),
        "confidence_summary": result["confidence_summary"],
        "confirmed_anomalies": result["anomalies"],
        "scan_quality": result["scan_quality"],
        "warnings": result["warnings"],
        "n_scans_processed": result["n_scans_processed"],
    }
    return JSONResponse(content=report)


# ─────────────────────────────────────────────────────────────────────────────
# Internal pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(session_id: str, files: list[str]) -> dict:
    """Full GMS pipeline: ingest → process → detect → validate → visualize."""

    # 1. Ingest
    datasets = ingestion_engine.load_multiple(files)
    if not datasets:
        raise ValueError("All scan files failed validation")

    grids = []
    detection_results = []
    heatmap_paths = {}

    for ds in datasets:
        # 2. Signal processing
        grid = signal_processor.process(ds)
        grids.append(grid)

        # 3. Anomaly detection
        detection = anomaly_detector.detect(grid)
        detection_results.append(detection)

        # 4. Visualize per scan
        paths = viz_engine.render_scan_heatmap(grid, detection,
                                               output_prefix=f"{session_id}_{grid.scan_id}")
        heatmap_paths[grid.scan_id] = paths

    # 5. Cross-scan validation + decision
    report = cross_validator.validate(detection_results, session_id=session_id)

    # 6. Final overview map
    viz_engine.render_final_report_map(grids, report,
                                        output_prefix=f"{session_id}_final")

    # Store heatmap paths in session
    sessions[session_id]["heatmap_paths"] = heatmap_paths

    # 7. Serialize to dict
    return _serialize_report(report)


def _serialize_report(report) -> dict:
    """Convert dataclass report to JSON-serializable dict."""
    confirmed = []
    for ca in report.confirmed_anomalies:
        confirmed.append({
            "group_id": ca.group_id,
            "centroid_x": ca.centroid_x,
            "centroid_y": ca.centroid_y,
            "label": ca.best_label,
            "scan_confirmations": ca.scan_confirmations,
            "contributing_scans": ca.contributing_scans,
            "combined_confidence": ca.combined_confidence,
            "mean_snr": ca.mean_snr,
            "mean_uncertainty": ca.mean_uncertainty,
            "label_agreement": ca.label_agreement,
            "spatial_consistency": ca.spatial_consistency,
        })

    single = []
    for a in report.single_detections:
        single.append({
            "anomaly_id": a.anomaly_id,
            "cx": round(a.cx, 2),
            "cy": round(a.cy, 2),
            "label": a.raw_label,
            "confidence": a.confidence,
            "snr": round(a.snr_robust, 2),
            "uncertainty": round(a.uncertainty, 3),
            "dipole_score": round(a.dipole_score, 3),
            "smoothness": round(a.smoothness_score, 3),
        })

    return {
        "decision": report.decision,
        "confidence_summary": report.confidence_summary,
        "anomalies": confirmed,
        "single_detections": single,
        "scan_quality": report.scan_quality,
        "warnings": report.warnings,
        "n_scans_processed": report.n_scans_processed,
    }


def _decision_rationale(result: dict) -> str:
    decision = result["decision"]
    n_confirmed = result["confidence_summary"].get("n_confirmed", 0)
    n_single = result["confidence_summary"].get("n_single", 0)
    overall = result["confidence_summary"].get("overall", 0)

    if decision == "DIG":
        return (
            f"DIG recommended: {n_confirmed} anomaly group(s) confirmed across "
            f"multiple independent scans with overall confidence {overall:.0%}. "
            f"Statistical evidence is sufficient for targeted investigation."
        )
    elif decision == "RESCAN":
        return (
            f"RESCAN recommended: Partial signal evidence detected "
            f"({n_confirmed} confirmed, {n_single} single-scan detections). "
            f"Insufficient multi-scan confirmation for DIG decision. "
            f"Additional scanning passes recommended."
        )
    else:
        return (
            "NO_DIG: No statistically significant anomalies detected above "
            "the robust noise threshold. Signal is consistent with background "
            "soil variation or measurement noise."
        )


def _get_complete_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(404, f"Session not found: {session_id}")
    session = sessions[session_id]
    if session["status"] != "complete":
        raise HTTPException(409, f"Session status: {session['status']} (not complete yet)")
    return session
