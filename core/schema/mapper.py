"""
GMS — SemanticFieldMapper  v2.3

Maps raw field names → semantic telemetry roles via alias tables.
Handles typos, vendor-specific names, and abbreviations.

The mapper NEVER fabricates a field. If no alias matches, the role
stays unmapped and the capability is marked unavailable.

Alias tables are extensible via device profiles (devices/*.yaml).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .inspector import FieldInventory

logger = logging.getLogger("gms.schema.mapper")


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL ROLE NAMES (internal semantic names used throughout the system)
# ─────────────────────────────────────────────────────────────────────────────
ROLES = [
    "signal",           # primary scalar field measurement
    "filtered_signal",  # pre-filtered version of signal
    "x",                # X position coordinate
    "y",                # Y position coordinate
    "heading",          # device heading / bearing (degrees)
    "baseline",         # baseline reference level
    "snr",              # signal-to-noise ratio
    "noise_floor",      # noise floor / EMI level
    "stability",        # device stability metric
    "timestamp",        # time index
    "quality",          # quality code / flag
    "speed",            # scan speed
    "altitude",         # Z / altitude (for aerial)
    "temperature",      # sensor temperature
]

# ─────────────────────────────────────────────────────────────────────────────
# ALIAS TABLE
# Maps every known vendor alias → canonical role name.
# All names should be lowercase (matching CSVInspector output).
# ─────────────────────────────────────────────────────────────────────────────
ALIAS_TABLE: dict[str, str] = {
    # signal
    "signal":             "signal",
    "value":              "signal",
    "val":                "signal",
    "grad":               "signal",
    "gradient":           "signal",
    "bt_value":           "signal",
    "magnetic":           "signal",
    "mag":                "signal",
    "field":              "signal",
    "sensor":             "signal",
    "measurement":        "signal",
    "raw":                "signal",
    "raw_signal":         "signal",
    "mv":                 "signal",
    "adc":                "signal",
    "reading":            "signal",
    "ch1":                "signal",
    "channel1":           "signal",
    "g":                  "signal",    # gradiometer shorthand

    # filtered_signal
    "filtered":           "filtered_signal",
    "filtered_signal":    "filtered_signal",
    "gradient_filtered":  "filtered_signal",
    "filt":               "filtered_signal",
    "smooth":             "filtered_signal",
    "smoothed":           "filtered_signal",
    "processed":          "filtered_signal",
    "corrected":          "filtered_signal",

    # x
    "x":                  "x",
    "xpos":               "x",
    "x_pos":              "x",
    "lon":                "x",
    "longitude":          "x",
    "easting":            "x",
    "east":               "x",
    "col":                "x",
    "column":             "x",
    "pos_x":              "x",

    # y
    "y":                  "y",
    "ypos":               "y",
    "y_pos":              "y",
    "lat":                "y",
    "latitude":           "y",
    "northing":           "y",
    "north":              "y",
    "row":                "y",
    "pos_y":              "y",

    # heading
    "heading":            "heading",
    "direction":          "heading",
    "bearing":            "heading",
    "azimuth":            "heading",
    "angle":              "heading",
    "yaw":                "heading",
    "orientation":        "heading",
    "compass":            "heading",
    "deg":                "heading",

    # baseline
    "baseline":           "baseline",
    "base":               "baseline",
    "reference":          "baseline",
    "ref":                "baseline",
    "dc_level":           "baseline",
    "background":         "baseline",
    "drift":              "baseline",
    "offset":             "baseline",

    # snr
    "snr":                "snr",
    "signal_to_noise":    "snr",
    "s_n":                "snr",
    "s/n":                "snr",
    "ratio":              "snr",
    "quality_ratio":      "snr",

    # noise_floor
    "noise":              "noise_floor",
    "noise_floor":        "noise_floor",
    "emi":                "noise_floor",
    "emi_level":          "noise_floor",
    "interference":       "noise_floor",
    "background_noise":   "noise_floor",
    "rms_noise":          "noise_floor",
    "noise_rms":          "noise_floor",

    # stability
    "stability":          "stability",
    "stable":             "stability",
    "quality_index":      "stability",
    "qi":                 "stability",
    "qf":                 "stability",
    "quality_flag":       "stability",
    "lock":               "stability",

    # timestamp
    "timestamp":          "timestamp",
    "time":               "timestamp",
    "t":                  "timestamp",
    "ts":                 "timestamp",
    "ms":                 "timestamp",
    "epoch":              "timestamp",
    "utc":                "timestamp",
    "datetime":           "timestamp",
    "sample":             "timestamp",
    "idx":                "timestamp",
    "index":              "timestamp",

    # quality
    "quality":            "quality",
    "note":               "quality",
    "notes":              "quality",
    "status":             "quality",
    "flag":               "quality",
    "valid":              "quality",
    "ok":                 "quality",

    # speed
    "speed":              "speed",
    "velocity":           "speed",
    "v":                  "speed",
    "scan_speed":         "speed",

    # altitude
    "altitude":           "altitude",
    "alt":                "altitude",
    "height":             "altitude",
    "z":                  "altitude",
    "elevation":          "altitude",
    "elev":               "altitude",

    # temperature
    "temperature":        "temperature",
    "temp":               "temperature",
    "celsius":            "temperature",
    "fahrenheit":         "temperature",
    "sensor_temp":        "temperature",
}


@dataclass
class RoleMapping:
    """One resolved semantic role → field name binding."""
    role: str                    # canonical role name
    field_name: Optional[str]    # actual column name in CSV (None if unmapped)
    confidence: float            # 1.0 = exact match, 0.7 = alias match
    is_mapped: bool

    def __bool__(self):
        return self.is_mapped


@dataclass
class SemanticMapping:
    """Complete mapping result: role → column name for this CSV."""
    source: str
    role_map: dict[str, RoleMapping]   # role → RoleMapping
    unmapped_fields: list[str]         # fields with no semantic role
    warnings: list[str] = field(default_factory=list)

    def get_column(self, role: str) -> Optional[str]:
        """Return the actual column name for a role, or None."""
        rm = self.role_map.get(role)
        return rm.field_name if rm and rm.is_mapped else None

    def has_role(self, role: str) -> bool:
        rm = self.role_map.get(role)
        return rm is not None and rm.is_mapped

    def mapped_roles(self) -> list[str]:
        return [r for r, rm in self.role_map.items() if rm.is_mapped]


class SemanticFieldMapper:
    """
    Maps raw CSV field names to canonical semantic roles.

    Rules:
    1. Exact match on canonical name → confidence 1.0
    2. Alias table match → confidence 0.85
    3. Prefix/suffix heuristic → confidence 0.60
    4. No match → unmapped (role stays None)

    Extra aliases can be injected at construction time from device profiles.
    """

    def __init__(self, extra_aliases: Optional[dict[str, str]] = None):
        self._aliases = dict(ALIAS_TABLE)
        if extra_aliases:
            self._aliases.update({k.lower(): v for k, v in extra_aliases.items()})

    def map(self, inventory: FieldInventory) -> SemanticMapping:
        """Resolve semantic roles for all fields in a FieldInventory."""
        warnings: list[str] = []
        field_names = inventory.field_names()

        # Build a reverse lookup: which field_names are already claimed
        claimed: dict[str, str] = {}   # field_name → role
        role_map: dict[str, RoleMapping] = {}

        for role in ROLES:
            result = self._resolve_role(role, field_names, claimed)
            if result.is_mapped:
                claimed[result.field_name] = role
            role_map[role] = result

        # Warn about any unmapped but likely-numeric fields
        unmapped = [f for f in field_names if f not in claimed]
        for f in unmapped:
            fi = inventory.get(f)
            if fi and fi.vmin is not None and not fi.is_likely_index:
                warnings.append(
                    f"Field '{f}' is numeric but has no recognized semantic role — "
                    f"it will not be used. Add it to the alias table or device profile."
                )

        logger.info(
            f"[SemanticFieldMapper] {inventory.source}: "
            f"mapped {len([r for r in role_map.values() if r.is_mapped])}/{len(ROLES)} roles. "
            f"Mapped: {[r for r, rm in role_map.items() if rm.is_mapped]}"
        )

        return SemanticMapping(
            source=inventory.source,
            role_map=role_map,
            unmapped_fields=unmapped,
            warnings=warnings,
        )

    def _resolve_role(
        self,
        role: str,
        field_names: list[str],
        claimed: dict[str, str],
    ) -> RoleMapping:
        """Try to bind a role to one of the available field names."""
        available = [f for f in field_names if f not in claimed]

        # Pass 1: exact canonical name match
        if role in available:
            return RoleMapping(role=role, field_name=role, confidence=1.0, is_mapped=True)

        # Pass 2: alias table match
        for alias, canonical in self._aliases.items():
            if canonical == role and alias in available:
                return RoleMapping(role=role, field_name=alias, confidence=0.85, is_mapped=True)

        # Pass 3: prefix/suffix heuristic
        for fname in available:
            if fname.startswith(role + "_") or fname.endswith("_" + role):
                return RoleMapping(role=role, field_name=fname, confidence=0.60, is_mapped=True)

        return RoleMapping(role=role, field_name=None, confidence=0.0, is_mapped=False)
