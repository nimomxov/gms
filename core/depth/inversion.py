"""
GMS — Depth Inversion Architecture Stub  v2.2

IMPORTANT: Depth estimation is DISABLED by default and by design.

This module provides the architecture interface for a future
physics-based depth inversion system. It does NOT estimate depth.

WHY depth is not implemented:
  1. Magnetic dipole depth estimation requires:
     - Known sensor height above ground (calibrated)
     - Known sensor spacing for gradiometer
     - Known magnetic inclination for the survey location
     - Known background field for the survey date/location
     - Multiple independent measurement directions
  2. Without calibration, ANY depth estimate is fabrication.
  3. Rule of thumb (Peters' half-width method, Nabighian etc.)
     are heuristics valid ONLY under specific assumptions that
     cannot be guaranteed without ground-truth validation.

Future implementation path:
  1. Add sensor geometry calibration (height, spacing, inclination)
  2. Implement forward model for vertical dipole
  3. Implement Euler deconvolution or Peters' half-width
  4. Validate against controlled burial experiments
  5. Add uncertainty bounds from inversion residuals
  6. Only then enable DepthInversionPlugin

Reference:
  Nabighian et al., "The historical development of the magnetic method
  in exploration," Geophysics, 70(6), 2005.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("gms.depth")

DEPTH_DISABLED_MESSAGE = (
    "Depth estimation unavailable — requires calibrated inversion. "
    "To enable: provide sensor_height_m, sensor_spacing_m, "
    "magnetic_inclination_deg in calibration config."
)


@dataclass
class SensorCalibration:
    """
    Required calibration data for depth inversion.
    All fields must be measured/known before depth estimation is valid.
    """
    sensor_height_m: float = None       # height above ground surface
    sensor_spacing_m: float = None      # gradiometer sensor spacing
    magnetic_inclination_deg: float = None  # local magnetic field inclination
    background_field_nT: float = None   # regional background field
    scan_speed_ms: float = None         # acquisition speed (for timing artifacts)
    calibrated: bool = False            # must be True to enable depth

    def validate(self) -> tuple[bool, str]:
        """Check if calibration is sufficient for depth estimation."""
        if not self.calibrated:
            return False, "calibrated=False — set to True after measuring all parameters"
        required = [self.sensor_height_m, self.sensor_spacing_m,
                    self.magnetic_inclination_deg]
        if any(v is None for v in required):
            missing = [n for n, v in [
                ("sensor_height_m", self.sensor_height_m),
                ("sensor_spacing_m", self.sensor_spacing_m),
                ("magnetic_inclination_deg", self.magnetic_inclination_deg),
            ] if v is None]
            return False, f"Missing calibration fields: {missing}"
        return True, "calibration valid"


class DepthInversionPlugin:
    """
    Architecture stub for future depth estimation.

    Currently returns DEPTH_DISABLED_MESSAGE for all inputs.
    When calibration is provided and validated, this class will be
    extended with actual inversion methods.

    Do NOT attempt to estimate depth from uncalibrated data.
    """

    def __init__(self, calibration: SensorCalibration = None):
        self.calibration = calibration or SensorCalibration()
        valid, msg = self.calibration.validate()
        if not valid:
            logger.debug(f"  DepthInversionPlugin: disabled ({msg})")
        self._enabled = valid

    @property
    def enabled(self) -> bool:
        return self._enabled

    def estimate_depth(self, anomaly) -> dict:
        """
        Attempt depth estimation for an anomaly.
        Returns dict with 'depth_m': None and explanation if not calibrated.
        """
        if not self._enabled:
            return {
                "depth_m": None,
                "depth_uncertainty_m": None,
                "method": None,
                "message": DEPTH_DISABLED_MESSAGE,
                "calibration_required": [
                    "sensor_height_m",
                    "sensor_spacing_m",
                    "magnetic_inclination_deg",
                ],
            }

        # Future implementation placeholder
        # When calibration is available, implement:
        #   1. Peters' half-width method for quick estimate
        #   2. Euler deconvolution for structural index
        #   3. Systematic uncertainty from inversion residuals
        return {
            "depth_m": None,
            "depth_uncertainty_m": None,
            "method": "pending_implementation",
            "message": (
                "Calibration received but inversion not yet implemented. "
                "Scheduled for GMS v3.0."
            ),
        }

    def add_to_report(self, report_dict: dict) -> dict:
        """Add depth section to a report dict. Always honest about status."""
        report_dict["depth_estimation"] = {
            "status": "disabled" if not self._enabled else "calibrated_pending",
            "message": DEPTH_DISABLED_MESSAGE,
            "enabled": self._enabled,
        }
        return report_dict
