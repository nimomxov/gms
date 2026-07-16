"""
GMS — Device Profile System  v2.3

Loads device profiles from devices/*.yaml and applies them to the
SemanticFieldMapper, overriding or extending alias tables.

If no profile matches the detected fields, Auto Detection Mode is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("gms.devices")


@dataclass
class DeviceProfile:
    """Parsed device profile from YAML."""
    name: str
    description: str
    field_aliases: dict[str, str]       # raw_name → semantic_role
    scaling: dict[str, float]           # role → multiplier
    value_range: list[float]            # [min, max] for signal
    telemetry_notes: str
    recommended_preset: str
    auto_detect_fields: list[str]       # fields that identify this device


class DeviceProfileRegistry:
    """
    Loads all device profiles and selects the best match for a field set.
    Falls back to generic auto-detection if no profile matches.
    """

    def __init__(self, profiles_dir: Optional[str | Path] = None):
        if profiles_dir is None:
            profiles_dir = Path(__file__).parent.parent / "devices"
        self.profiles_dir = Path(profiles_dir)
        self._profiles: dict[str, DeviceProfile] = {}
        self._load_all()

    def _load_all(self):
        if not self.profiles_dir.exists():
            logger.warning(f"Device profiles directory not found: {self.profiles_dir}")
            return
        for yaml_file in self.profiles_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                profile = DeviceProfile(
                    name=data.get("name", yaml_file.stem),
                    description=data.get("description", ""),
                    field_aliases=data.get("field_aliases", {}),
                    scaling=data.get("scaling", {}),
                    value_range=data.get("value_range", [0, 65535]),
                    telemetry_notes=data.get("telemetry_notes", ""),
                    recommended_preset=data.get("recommended_preset", "stable"),
                    auto_detect_fields=data.get("auto_detect_fields", []),
                )
                self._profiles[profile.name] = profile
                logger.debug(f"Loaded device profile: {profile.name}")
            except Exception as e:
                logger.warning(f"Failed to load profile {yaml_file.name}: {e}")

    def detect(self, field_names: list[str]) -> Optional[DeviceProfile]:
        """
        Find the best matching profile based on detected field names.
        Returns None if no profile matches (triggers Auto Detection Mode).
        """
        best = None
        best_score = 0
        lowered = [f.lower() for f in field_names]

        for profile in self._profiles.values():
            if not profile.auto_detect_fields:
                continue
            matches = sum(
                1 for f in profile.auto_detect_fields if f.lower() in lowered
            )
            score = matches / len(profile.auto_detect_fields)
            if score > best_score:
                best_score = score
                best = profile

        if best and best_score >= 0.6:
            logger.info(
                f"[DeviceProfileRegistry] Matched profile '{best.name}' "
                f"(score={best_score:.0%})"
            )
            return best

        logger.info(
            "[DeviceProfileRegistry] No profile matched — using Auto Detection Mode"
        )
        return None

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())
