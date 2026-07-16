"""
GMS — Processing Session  v1.0
================================
Full provenance graph for every analysis run.

Every pipeline execution produces a ProcessingSession that captures:
  - all inputs (file hashes, capability fingerprint)
  - all stage parameters (interpolator, baseline, detector configs)
  - all stage timings
  - all warnings and disabled features
  - final decision with supporting evidence
  - config hash for exact reproduction

Why this matters:
  - Scientific reproducibility: given the same session JSON, any
    researcher can recreate the exact result.
  - Debugging: tracing exactly which stage introduced an artefact.
  - ML training: each session is a labeled training sample.
  - Audit trail: field teams can attach session JSON to dig reports.

Usage:
    session = ProcessingSession.begin(scan_files, preset, capabilities)
    session.record_stage("interpolation", params={...}, duration_ms=120)
    session.record_warning("baseline", "high residual energy")
    session.finalize(result_dict)
    session.export_json("reports/session_abc123.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("gms.session")


# ─────────────────────────────────────────────────────────────────────────────
# Stage record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    name: str
    status: str                    # completed | failed | skipped
    params: dict = field(default_factory=dict)
    duration_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    output_hash: str = ""          # SHA-256 of stage output array (reproducibility)


# ─────────────────────────────────────────────────────────────────────────────
# Capability snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CapabilitySnapshot:
    grade: str
    has_position: bool
    has_snr: bool
    has_heading: bool
    has_baseline: bool
    column_map: dict
    enabled_stages: list[str]
    disabled_stages: dict          # stage → reason
    device_profile: str
    n_samples: int

    @classmethod
    def from_dataset(cls, dataset) -> "CapabilitySnapshot":
        cap = dataset.capabilities
        pipeline = dataset.pipeline
        return cls(
            grade=cap.grade.name,
            has_position=cap.has_position,
            has_snr=cap.has_snr,
            has_heading=cap.has_heading,
            has_baseline=getattr(cap, "has_baseline", False),
            column_map=dict(cap.column_map),
            enabled_stages=list(getattr(pipeline, "enabled_stages", [])),
            disabled_stages=dict(getattr(pipeline, "disabled_stages", {})),
            device_profile=getattr(dataset, "device_profile_name", "unknown") or "unknown",
            n_samples=dataset.n_samples,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ProcessingSession
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessingSession:
    """
    Full provenance record for one analysis run.
    Immutable once finalized.
    """

    session_id: str
    started_at: str                        # ISO-8601 UTC
    finished_at: str = ""
    gms_version: str = "3.1"

    # Inputs
    scan_files: list[str] = field(default_factory=list)
    dataset_hashes: dict[str, str] = field(default_factory=dict)   # file → sha256
    preset: str = "stable"
    pipeline_hash: str = ""                # PipelineConfig.config_hash()
    telemetry_grade: str = "UNKNOWN"
    capability_snapshot: Optional[CapabilitySnapshot] = None

    # Stages
    stages: list[StageRecord] = field(default_factory=list)

    # Outputs
    final_decision: str = ""
    n_confirmed_anomalies: int = 0
    overall_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    result_summary: dict = field(default_factory=dict)

    # Flags
    finalized: bool = False
    reproducible: bool = True              # False if non-deterministic stage ran

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def begin(
        cls,
        scan_files: list[str],
        preset: str,
        dataset=None,
        pipeline_config=None,
    ) -> "ProcessingSession":
        session_id = _make_session_id(scan_files, preset)
        dataset_hashes = {f: _file_hash(f) for f in scan_files}

        cap_snap = None
        if dataset is not None:
            try:
                cap_snap = CapabilitySnapshot.from_dataset(dataset)
            except Exception as e:
                logger.warning(f"[Session] Could not snapshot capabilities: {e}")

        pipeline_hash = ""
        if pipeline_config is not None:
            try:
                pipeline_hash = pipeline_config.config_hash()
            except Exception:
                pass

        grade = "UNKNOWN"
        if dataset is not None:
            grade = getattr(getattr(dataset, "grade", None), "name", "UNKNOWN")

        session = cls(
            session_id=session_id,
            started_at=_utcnow(),
            scan_files=scan_files,
            dataset_hashes=dataset_hashes,
            preset=preset,
            pipeline_hash=pipeline_hash,
            telemetry_grade=grade,
            capability_snapshot=cap_snap,
        )
        logger.info(f"[Session] Started: {session_id}")
        return session

    # ── Stage recording ────────────────────────────────────────────────────

    def record_stage(
        self,
        name: str,
        status: str = "completed",
        params: dict = None,
        duration_ms: int = 0,
        warnings: list[str] = None,
        error: str = "",
        output_array=None,          # numpy array → hashed for reproducibility
    ):
        output_hash = ""
        if output_array is not None:
            try:
                import numpy as np
                arr = np.asarray(output_array).ravel()[:10000]   # sample
                output_hash = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
            except Exception:
                pass

        record = StageRecord(
            name=name,
            status=status,
            params=params or {},
            duration_ms=duration_ms,
            warnings=warnings or [],
            error=error,
            output_hash=output_hash,
        )
        self.stages.append(record)
        logger.debug(f"[Session] Stage recorded: {name} ({status}) {duration_ms}ms")

    def record_warning(self, stage: str, warning: str):
        full = f"[{stage}] {warning}"
        self.warnings.append(full)
        # Also attach to last stage with matching name
        for s in reversed(self.stages):
            if s.name == stage:
                s.warnings.append(warning)
                break

    # ── Finalization ───────────────────────────────────────────────────────

    def finalize(self, result: dict):
        """
        Seal the session with pipeline results.
        After this call, export_json() is safe.
        """
        self.finished_at = _utcnow()
        self.final_decision = result.get("decision", "UNKNOWN")
        confirmed = result.get("confirmed_anomalies", [])
        self.n_confirmed_anomalies = len(confirmed)

        conf_summary = result.get("confidence_summary", {})
        self.overall_confidence = conf_summary.get("overall", 0.0)

        # Warnings from pipeline result
        for w in result.get("warnings", []):
            if w not in self.warnings:
                self.warnings.append(w)

        # Lightweight summary (serialization-safe)
        self.result_summary = {
            "decision": self.final_decision,
            "n_confirmed": self.n_confirmed_anomalies,
            "overall_confidence": round(self.overall_confidence, 3),
            "n_scans": result.get("n_scans_processed", 0),
            "pipeline": result.get("pipeline", {}),
        }

        self.finalized = True
        logger.info(
            f"[Session] Finalized: {self.session_id} → "
            f"{self.final_decision} ({self.n_confirmed_anomalies} confirmed)"
        )

    # ── Export ────────────────────────────────────────────────────────────

    def export_json(self, path: str = None) -> str:
        """Export full provenance JSON. Returns the JSON string."""
        if path is None:
            path = f"reports/{self.session_id}_provenance.json"

        data = {
            "gms_version": self.gms_version,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "preset": self.preset,
            "pipeline_hash": self.pipeline_hash,
            "telemetry_grade": self.telemetry_grade,
            "scan_files": self.scan_files,
            "dataset_hashes": self.dataset_hashes,
            "capability_snapshot": (
                asdict(self.capability_snapshot)
                if self.capability_snapshot else None
            ),
            "stages": [asdict(s) for s in self.stages],
            "warnings": self.warnings,
            "result": self.result_summary,
            "reproducible": self.reproducible,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(data, indent=2, default=str)
        Path(path).write_text(json_str, encoding="utf-8")
        logger.info(f"[Session] Provenance exported: {path}")
        return json_str

    def total_duration_ms(self) -> int:
        return sum(s.duration_ms for s in self.stages)

    def stage_summary(self) -> list[dict]:
        return [
            {
                "stage": s.name,
                "status": s.status,
                "ms": s.duration_ms,
                "warnings": len(s.warnings),
            }
            for s in self.stages
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Session registry (in-memory; survives single app session)
# ─────────────────────────────────────────────────────────────────────────────

class SessionRegistry:
    """Keeps the last N sessions in memory for the current app run."""

    MAX_SESSIONS = 20
    _sessions: list[ProcessingSession] = []

    @classmethod
    def register(cls, session: ProcessingSession):
        cls._sessions.append(session)
        if len(cls._sessions) > cls.MAX_SESSIONS:
            cls._sessions.pop(0)

    @classmethod
    def latest(cls) -> Optional[ProcessingSession]:
        return cls._sessions[-1] if cls._sessions else None

    @classmethod
    def all(cls) -> list[ProcessingSession]:
        return list(cls._sessions)

    @classmethod
    def find(cls, session_id: str) -> Optional[ProcessingSession]:
        for s in cls._sessions:
            if s.session_id == session_id:
                return s
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(filepath: str) -> str:
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "unavailable"


def _make_session_id(scan_files: list[str], preset: str) -> str:
    key = "|".join(sorted(scan_files)) + "|" + preset
    return "s_" + hashlib.sha256(key.encode()).hexdigest()[:12]
