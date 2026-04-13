"""Ramachandran reference densities for FACET visualization.

Provides per-amino-acid Ramachandran density grids and helpers for the
interactive inspector. Densities are synthesized from a mixture of
Gaussians representing the canonical secondary-structure regions
(alpha-R helix, beta-strand, polyproline II, alpha-L helix, bridge),
with per-AA weights tuned from public PDB-wide statistics.

This is bundled reference data — no external files or training-corpus
dependency. Densities are computed once per AA and cached.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# 5-degree resolution grid over [-180, 180] in both phi and psi.
GRID_RES = 5
GRID_N = 360 // GRID_RES + 1  # 73 cells inclusive
GRID_PHI = np.linspace(-180, 180, GRID_N)
GRID_PSI = np.linspace(-180, 180, GRID_N)

# Canonical Ramachandran components: (mu_phi, mu_psi, sigma_phi, sigma_psi, rho)
_ALPHA_R    = (-63.0,  -42.0, 8.0,  9.0,   0.10)
_ALPHA_R_W  = (-72.0,  -25.0, 12.0, 14.0,  0.05)
_BETA       = (-118.0, 130.0, 14.0, 14.0, -0.10)
_BETA_W     = (-138.0, 155.0, 18.0, 16.0,  0.00)
_PPII       = (-75.0,  145.0, 9.0,  12.0,  0.00)
_ALPHA_L    = ( 60.0,   45.0, 9.0,  10.0,  0.00)
_BRIDGE     = (-90.0,    0.0, 14.0, 14.0,  0.00)

_COMPONENTS = (_ALPHA_R, _ALPHA_R_W, _BETA, _BETA_W, _PPII, _ALPHA_L, _BRIDGE)

# Default per-component weights (most amino acids).
_DEFAULT_W = (0.35, 0.10, 0.20, 0.10, 0.18, 0.02, 0.05)

# Per-AA tweaks reflecting known biochemistry. Order matches _COMPONENTS.
_AA_WEIGHTS = {
    "GLY": (0.10, 0.05, 0.15, 0.10, 0.20, 0.30, 0.10),  # bilateral, alpha-L common
    "PRO": (0.40, 0.05, 0.05, 0.05, 0.40, 0.00, 0.05),  # ring-constrained, ppII strong
    "VAL": (0.20, 0.05, 0.35, 0.20, 0.15, 0.00, 0.05),  # branched: more strand
    "ILE": (0.20, 0.05, 0.35, 0.20, 0.15, 0.00, 0.05),  # branched: more strand
    "THR": (0.25, 0.10, 0.25, 0.15, 0.20, 0.00, 0.05),  # branched
    "ALA": (0.45, 0.15, 0.15, 0.05, 0.15, 0.02, 0.03),  # helix-loving
    "GLU": (0.45, 0.15, 0.15, 0.05, 0.15, 0.02, 0.03),
    "LEU": (0.40, 0.15, 0.20, 0.05, 0.15, 0.02, 0.03),
    "MET": (0.40, 0.15, 0.20, 0.05, 0.15, 0.02, 0.03),
    "LYS": (0.40, 0.15, 0.20, 0.05, 0.15, 0.02, 0.03),
    "ARG": (0.40, 0.15, 0.20, 0.05, 0.15, 0.02, 0.03),
    "GLN": (0.40, 0.15, 0.20, 0.05, 0.15, 0.02, 0.03),
    "ASN": (0.30, 0.10, 0.15, 0.05, 0.25, 0.10, 0.05),  # turn-loving
    "ASP": (0.30, 0.10, 0.15, 0.05, 0.25, 0.10, 0.05),  # turn-loving
    "SER": (0.30, 0.10, 0.20, 0.10, 0.20, 0.05, 0.05),
    "HIS": (0.32, 0.10, 0.22, 0.10, 0.20, 0.03, 0.03),
    "CYS": (0.30, 0.10, 0.25, 0.10, 0.20, 0.02, 0.03),
    "PHE": (0.30, 0.10, 0.25, 0.15, 0.15, 0.02, 0.03),
    "TYR": (0.30, 0.10, 0.25, 0.15, 0.15, 0.02, 0.03),
    "TRP": (0.32, 0.10, 0.25, 0.13, 0.15, 0.02, 0.03),
}

CANONICAL_AA = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)


def _gaussian_2d(phi_grid, psi_grid, mean_phi, mean_psi, sigma_phi, sigma_psi, rho):
    # Periodic distance: closest image across the [-180, 180] wrap.
    dphi = (phi_grid - mean_phi + 180.0) % 360.0 - 180.0
    dpsi = (psi_grid - mean_psi + 180.0) % 360.0 - 180.0
    z = (
        (dphi / sigma_phi) ** 2
        - 2 * rho * (dphi / sigma_phi) * (dpsi / sigma_psi)
        + (dpsi / sigma_psi) ** 2
    )
    norm = 2 * np.pi * sigma_phi * sigma_psi * np.sqrt(1 - rho ** 2)
    return np.exp(-z / (2 * (1 - rho ** 2))) / norm


def _build_density(weights):
    phi_g, psi_g = np.meshgrid(GRID_PHI, GRID_PSI, indexing="xy")
    density = np.zeros_like(phi_g)
    for w, comp in zip(weights, _COMPONENTS):
        if w <= 0:
            continue
        density += w * _gaussian_2d(phi_g, psi_g, *comp)
    return density / density.max()


@lru_cache(maxsize=32)
def get_density(aa_3letter: str) -> np.ndarray:
    """Per-AA Ramachandran density grid, shape (GRID_N, GRID_N)."""
    weights = _AA_WEIGHTS.get(aa_3letter.upper(), _DEFAULT_W)
    return _build_density(weights)


@lru_cache(maxsize=1)
def get_global_density() -> np.ndarray:
    """Global all-AA density for the Tier-4 background, uniform AA weighting."""
    grids = [get_density(aa) for aa in CANONICAL_AA]
    return np.mean(grids, axis=0)


def density_contour_levels(density: np.ndarray) -> list[float]:
    """Density thresholds at the 99%, 90%, 50% cumulative mass quantiles.

    Returns levels in ascending order: [99% mass, 90% mass, 50% mass].
    """
    vals_sorted = np.sort(density.flatten())[::-1]
    cumsum = np.cumsum(vals_sorted)
    total = cumsum[-1]
    levels = []
    for q in (0.99, 0.90, 0.50):
        idx = int(np.searchsorted(cumsum, q * total))
        idx = min(idx, len(vals_sorted) - 1)
        levels.append(float(vals_sorted[idx]))
    return sorted(levels)


# Canonical Ramachandran zones for human-readable region tagging.
# (label, mean_phi, mean_psi, capture_radius_deg)
_REGIONS = [
    ("alpha-helix (right)",  -63.0, -42.0, 35.0),
    ("alpha-helix (right, wide)", -72.0, -25.0, 35.0),
    ("beta-strand",         -120.0, 130.0, 40.0),
    ("polyproline II",       -75.0, 145.0, 25.0),
    ("alpha-helix (left)",    60.0,  45.0, 30.0),
    ("bridge / 3-10 helix",  -90.0,   0.0, 30.0),
]


def classify_region(phi: float, psi: float) -> str:
    """Closest canonical Ramachandran zone label for a (phi, psi) point."""
    best_label = "uncommon region"
    best_dist = float("inf")
    for label, mu_phi, mu_psi, radius in _REGIONS:
        dphi = (phi - mu_phi + 180.0) % 360.0 - 180.0
        dpsi = (psi - mu_psi + 180.0) % 360.0 - 180.0
        d = float(np.sqrt(dphi * dphi + dpsi * dpsi))
        if d < radius and d < best_dist:
            best_dist = d
            best_label = label
    return best_label


def classify_outlier(phi: float, psi: float, aa_3letter: str) -> str:
    """Label the (phi, psi) location relative to the AA's density quantiles.

    Returns one of: "core", "allowed", "marginal", "outlier".
    """
    density = get_density(aa_3letter)
    levels = density_contour_levels(density)  # [99% mass, 90% mass, 50% mass]
    phi_idx = int(round((phi + 180) / GRID_RES))
    psi_idx = int(round((psi + 180) / GRID_RES))
    phi_idx = max(0, min(GRID_N - 1, phi_idx))
    psi_idx = max(0, min(GRID_N - 1, psi_idx))
    val = float(density[psi_idx, phi_idx])
    if val >= levels[2]:
        return "core"
    if val >= levels[1]:
        return "allowed"
    if val >= levels[0]:
        return "marginal"
    return "outlier"
