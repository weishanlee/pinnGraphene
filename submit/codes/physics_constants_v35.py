#!/usr/bin/env python3
"""
Physics constants for SCMS-PINN v34
Preserved from v33 - maintaining all TB-related constants
"""

import numpy as np

# Graphene lattice parameters
LATTICE_CONSTANT = 2.46  # Angstroms (graphene lattice constant)
BOND_LENGTH = LATTICE_CONSTANT / np.sqrt(3)  # C-C bond length = a/√3 ≈ 1.42 Å

# Tight-binding parameters
HOPPING_PARAMETER = 2.7  # eV (nearest-neighbor hopping)

# Brillouin zone
K_POINT_MAGNITUDE = 4 * np.pi / (3 * LATTICE_CONSTANT)  # |K| in reciprocal space

# Physical constants
EXPECTED_FERMI_VELOCITY = 3 * HOPPING_PARAMETER * LATTICE_CONSTANT / (2 * np.sqrt(3))  # eV·Å

# Training hyperparameters
DELTA_K = 0.01  # Small k-space displacement for finite differences
FERMI_VELOCITY_RADIUS_MIN = 0.02  # Inner radius for Fermi velocity calculation
FERMI_VELOCITY_RADIUS_MAX = 0.2   # Outer radius for Fermi velocity calculation

# Smoothness loss parameters (v33: will be disabled or reduced)
SMOOTHNESS_RADIUS = 0.05  # Radius for smoothness calculation
SMOOTHNESS_WEIGHT = 0.05  # Initial weight for smoothness loss

# Plotting parameters
PLOT_K_RANGE = 3.5  # Range for k-space plots in Å^(-1)
PLOT_ENERGY_RANGE = 10  # Energy range for band structure plots in eV

# Critical points for validation
CRITICAL_POINTS = {
    'gamma': {'energy': 3 * HOPPING_PARAMETER, 'gap': 6 * HOPPING_PARAMETER},
    'K': {'energy': 0.0, 'gap': 0.0},
    'M': {'energy': HOPPING_PARAMETER, 'gap': 2 * HOPPING_PARAMETER}
}

print(f"Physics constants loaded for v34:")
print(f"  Lattice constant: {LATTICE_CONSTANT} Å")
print(f"  Bond length: {BOND_LENGTH:.3f} Å")
print(f"  Hopping parameter: {HOPPING_PARAMETER} eV")
print(f"  Expected Fermi velocity: {EXPECTED_FERMI_VELOCITY:.3f} eV·Å")