"""
GMS — Topology Validator  v2.3

Problem solved:
  rock_debris_clutter → DIG  (should be RESCAN)
  Debris fields produce complex multi-peak blobs that don't match dipole topology.
  A true buried ferrous target produces a COMPACT, SYMMETRIC, ORGANIZED pattern.

Shape descriptors computed:
  - Compactness    : 4π·area / perimeter²  (circle=1, elongated<1, fragmented<<1)
  - Eccentricity   : ratio of eigenvalues of covariance matrix (0=round, 1=line)
  - Lobe symmetry  : how symmetric the blob is around its centroid
  - Fragmentation  : number of significant sub-blobs within the candidate region
  - Signal entropy : Shannon entropy of amplitude histogram (organized=low, chaotic=high)
  - Peak isolation : is the peak cell significantly above its neighbors?

Physical basis:
  A buried ferrous dipole at depth z produces a smooth, compact pattern.
  Rock debris at the surface produces sharp, fragmented, multi-peak patterns.
  Basalt formations produce very smooth, broad, non-compact patterns.

Integration:
  TopologyValidator is added as Stage 2.5 in the cascade (after DipoleValidator,
  before CoherenceValidator) — it operates on the blob geometry.
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger("gms.topology")


def _perimeter(mask: np.ndarray) -> float:
    """Approximate blob perimeter using boundary cell count."""
    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded
    return float(boundary.sum()) + 1e-3


def _compactness(mask: np.ndarray) -> float:
    """
    Compactness = 4π·area / perimeter²
    Circle: 1.0. Square: ~0.785. Long thin rod: → 0. Fragmented: → 0.
    """
    area = float(mask.sum())
    perim = _perimeter(mask)
    return float(np.clip(4 * np.pi * area / (perim**2), 0, 1))


def _eccentricity(mask: np.ndarray) -> float:
    """
    Eccentricity from covariance matrix eigenvalues.
    0 = perfectly circular. 1 = perfectly linear.
    """
    rows, cols = np.where(mask)
    if len(rows) < 4:
        return 0.0
    cy, cx = np.mean(rows), np.mean(cols)
    dr, dc = rows - cy, cols - cx
    cov = np.array([
        [np.mean(dr**2), np.mean(dr*dc)],
        [np.mean(dr*dc), np.mean(dc**2)],
    ])
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))[::-1]
        if eigvals[0] < 1e-8:
            return 0.0
        return float(np.clip(np.sqrt(1.0 - eigvals[1]/eigvals[0]), 0, 1))
    except Exception:
        return 0.0


def _lobe_symmetry(subgrid: np.ndarray) -> float:
    """
    Measures how symmetric the signal is around its centroid.
    Dipole: partially asymmetric but organized (0.4-0.7).
    Debris: highly asymmetric or chaotic (< 0.2 or > 0.9 with no structure).
    Returns: 0 = completely asymmetric, 1 = perfectly symmetric.
    """
    h, w = subgrid.shape
    if h < 3 or w < 3:
        return 1.0
    # Flip and correlate with itself
    flipped_h = np.flipud(subgrid)
    flipped_v = np.fliplr(subgrid)
    
    norm = np.linalg.norm(subgrid.ravel()) + 1e-8
    sym_h = float(np.dot(subgrid.ravel(), flipped_h.ravel())) / (norm**2)
    sym_v = float(np.dot(subgrid.ravel(), flipped_v.ravel())) / (norm**2)
    
    # Average symmetry across both axes
    return float(np.clip(0.5 * (sym_h + sym_v + 2) / 2, 0, 1))  # normalize to [0,1]


def _fragmentation(blob_mask: np.ndarray, subgrid: np.ndarray,
                    amplitude_threshold: float = 0.5) -> int:
    """
    Count significant sub-blobs within the candidate region.
    A true dipole: 1-2 lobes.
    Debris field: 3+ distinct peaks.
    """
    if subgrid.size == 0:
        return 0
    peak = float(np.abs(subgrid).max())
    if peak < 1e-6:
        return 0
    threshold = amplitude_threshold * peak
    significant = np.abs(subgrid) > threshold
    _, n_components = ndimage.label(significant)
    return n_components


def _signal_entropy(region: np.ndarray, n_bins: int = 16) -> float:
    """
    Shannon entropy of amplitude histogram.
    Organized signal (dipole): low entropy (concentrated distribution).
    Chaotic signal (debris/noise): high entropy (spread distribution).
    Returns: normalized to [0, 1] where 1 = max entropy.
    """
    if region.size < 4:
        return 0.5
    counts, _ = np.histogram(region, bins=n_bins)
    counts = counts[counts > 0]
    if len(counts) < 2:
        return 0.0
    max_entropy = np.log2(n_bins)
    e = float(scipy_entropy(counts, base=2))
    return float(np.clip(e / max_entropy, 0, 1))


def _peak_isolation(subgrid: np.ndarray, peak_radius: int = 2) -> float:
    """
    How isolated is the peak cell from its neighbors?
    A real compact target: peak >> surrounding cells.
    Broad basalt: peak ≈ surrounding cells.
    Returns: ratio of peak to local mean (clipped to [1, 10]).
    """
    if subgrid.size == 0:
        return 1.0
    peak_val = float(np.abs(subgrid).max())
    pr, pc = np.unravel_index(np.argmax(np.abs(subgrid)), subgrid.shape)
    
    r0 = max(0, pr - peak_radius); r1 = min(subgrid.shape[0], pr + peak_radius + 1)
    c0 = max(0, pc - peak_radius); c1 = min(subgrid.shape[1], pc + peak_radius + 1)
    neighborhood = subgrid[r0:r1, c0:c1]
    
    local_mean = float(np.mean(np.abs(neighborhood))) + 1e-6
    return float(np.clip(peak_val / local_mean, 1, 10))


class TopologyDescriptor:
    """All topology metrics for one blob candidate."""
    def __init__(self, blob_mask, subgrid, region):
        self.compactness   = _compactness(blob_mask)
        self.eccentricity  = _eccentricity(blob_mask)
        self.lobe_symmetry = _lobe_symmetry(subgrid)
        self.fragmentation = _fragmentation(blob_mask, subgrid)
        self.entropy       = _signal_entropy(region)
        self.peak_isolation = _peak_isolation(subgrid)

    def __repr__(self):
        return (
            f"Topology(compact={self.compactness:.2f}, "
            f"eccent={self.eccentricity:.2f}, "
            f"symm={self.lobe_symmetry:.2f}, "
            f"frags={self.fragmentation}, "
            f"entropy={self.entropy:.2f}, "
            f"isolation={self.peak_isolation:.1f})"
        )


class TopologyValidator:
    """
    Cascade Stage 2.5: Rejects physically implausible blob shapes.

    Rules (ALL must pass):
      1. Not too elongated (eccentricity < max_eccentricity)
      2. Not too fragmented (n_sub_blobs <= max_fragments)
      3. Not too chaotic (entropy < max_entropy)
      4. Has a discernible peak (peak_isolation >= min_peak_isolation)
      5. Minimum compactness (> min_compactness)

    What gets rejected:
      - Rock/debris clusters: high fragmentation + high entropy
      - Scan-line streaks: high eccentricity (already in CoherenceValidator, reinforced here)
      - Basalt: broad, low isolation, moderate compactness — but combined with
        EnvironmentalRejector size check → rejected
      - EMI spikes: tiny extent → filtered by min_spatial_extent before cascade

    What is preserved:
      - True ferrous dipoles: compact, 1-2 lobes, organized
      - Cavities: low compactness is OK for large smooth blobs → smoothness check guards
    """

    def __init__(self,
                 max_eccentricity: float = 0.92,
                 max_fragments: int = 6,
                 max_entropy: float = 0.88,
                 min_peak_isolation: float = 1.05,
                 min_compactness: float = 0.04,
                 max_extent_fraction: float = 0.60):
        self.max_eccentricity    = max_eccentricity
        self.max_fragments       = max_fragments
        self.max_entropy         = max_entropy
        self.min_peak_isolation  = min_peak_isolation
        self.min_compactness     = min_compactness
        self.max_extent_fraction = max_extent_fraction  # reject if blob covers >60% of grid

    def validate(self, blob_mask: np.ndarray,
                 subgrid: np.ndarray,
                 region: np.ndarray,
                 grid_size: int = None) -> tuple[bool, str, TopologyDescriptor]:
        """
        Returns: (passed, rejection_reason, descriptor)
        """
        td = TopologyDescriptor(blob_mask, subgrid, region)

        # Rule 0: Extent fraction — reject if blob covers most of the grid
        # (NCC fired everywhere — likely background, not a target)
        if grid_size is not None:
            extent_fraction = blob_mask.sum() / max(grid_size, 1)
            if extent_fraction > self.max_extent_fraction:
                return False, (
                    f"oversized: {extent_fraction:.0%} of grid (raise NCC threshold or check baseline)"
                ), td

        # Rule 1: Eccentricity — not a streak
        if td.eccentricity > self.max_eccentricity:
            return False, f"streak: eccentricity={td.eccentricity:.2f} > {self.max_eccentricity}", td

        # Rule 2: Fragmentation — not debris
        if td.fragmentation > self.max_fragments:
            return False, f"debris: {td.fragmentation} sub-blobs > {self.max_fragments}", td

        # Rule 3: Entropy — not chaotic
        if td.entropy > self.max_entropy:
            return False, f"chaotic: entropy={td.entropy:.2f} > {self.max_entropy}", td

        # Rule 4: Peak isolation — has real signal structure
        if td.peak_isolation < self.min_peak_isolation:
            return False, (
                f"flat: peak_isolation={td.peak_isolation:.2f} < {self.min_peak_isolation}"
            ), td

        # Rule 5: Compactness — not too fragmented
        if td.compactness < self.min_compactness:
            return False, f"fragmented: compactness={td.compactness:.3f} < {self.min_compactness}", td

        return True, "", td
