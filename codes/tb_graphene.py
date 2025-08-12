#!/usr/bin/env python3
"""
CORRECT TB utilities for graphene - Fixed version
This module contains the correct implementations based on standard graphene conventions
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List

# Physical constants
LATTICE_CONSTANT = 2.46  # Angstroms (graphene lattice constant)
HOPPING_PARAMETER = 2.7  # eV (graphene hopping parameter)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_reciprocal_basis() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reciprocal lattice vectors for graphene.
    
    For real space primitive vectors:
    a1 = (√3a/2, a/2)
    a2 = (√3a/2, -a/2)
    
    The reciprocal vectors are:
    b1 = (2π/√3a, 2π/a)
    b2 = (2π/√3a, -2π/a)
    
    These satisfy: a_i · b_j = 2π δ_ij
    
    Returns:
        b1, b2: Reciprocal lattice vectors
    """
    a = LATTICE_CONSTANT
    b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
    b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
    return b1, b2


class TightBindingTarget:
    """
    Tight-binding model for graphene following v27/v28 convention
    
    Convention:
    - a = lattice constant = 2.46 Å
    - d_CC = bond length = nearest-neighbor distance = a/√3 = 1.42 Å
    
    The TB formula uses bond length d_CC:
    |f(k)|² = 3 + 2cos(√3 ky d_CC) + 4cos(√3 ky d_CC/2)cos(3 kx d_CC/2)
    
    This gives:
    - Zero gap at K-points
    - ±3t at Γ point  
    - ±t at M points (Van Hove singularity)
    """
    def __init__(self):
        self.a = LATTICE_CONSTANT  # Lattice constant = 2.46 Å
        self.d_CC = LATTICE_CONSTANT / np.sqrt(3)  # Bond length = a/√3 = 1.42 Å
        self.t = HOPPING_PARAMETER  # Hopping parameter = 2.7 eV
        
    def __call__(self, k_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate TB band structure
        k_points: (N, 2) tensor of kx, ky values
        """
        kx = k_points[:, 0]
        ky = k_points[:, 1]
        
        # TB formula using bond length d_CC (following v27/v28)
        d_CC = self.d_CC
        
        term1 = 3.0
        term2 = 2 * torch.cos(np.sqrt(3) * ky * d_CC)
        term3 = 4 * torch.cos(np.sqrt(3) * ky * d_CC / 2) * torch.cos(3 * kx * d_CC / 2)
        
        # Calculate gamma
        gamma = term1 + term2 + term3
        
        # Ensure non-negative (use small epsilon for numerical stability)
        gamma = torch.clamp_min(gamma, 1e-12)
        
        # Calculate energies
        f_k = torch.sqrt(gamma)
        valence = -self.t * f_k
        conduction = self.t * f_k
        gap = conduction - valence
        
        # Stack energies
        energies = torch.stack([valence, conduction], dim=-1)
        
        return {
            'valence': valence,
            'conduction': conduction,
            'gap': gap,
            'energies': energies
        }


def generate_k_path(n_points: int = 1000) -> Tuple[torch.Tensor, List[str], List[int]]:
    """
    Generate k-points along high-symmetry path following v27/v28 convention.
    
    Path: Γ-Σ-M-K-Λ-Γ
    
    Uses reciprocal lattice vectors to define high-symmetry points correctly.
    
    Returns:
        k_path: Tensor of k-points
        labels: List of high-symmetry point labels
        positions: List of positions in the path
    """
    # Get correct reciprocal basis
    b1, b2 = get_reciprocal_basis()
    
    # High-symmetry points
    gamma = np.array([0, 0])
    # M point at edge center of BZ (there are 3 equivalent M points)
    M = 0.5 * (b1 + b2)  # M at (1/2, 1/2) in reciprocal coordinates
    # K point at corner of BZ (Dirac point)
    K = (2*b1 + b2) / 3  # K at (2/3, 1/3) in reciprocal coordinates
    
    # Additional points
    sigma = (gamma + M) / 2  # Halfway between Γ and M
    lambda_point = (K + gamma) / 2  # Halfway between K and Γ
    
    # Path segments: Γ-Σ-M-K-Λ-Γ
    segments = [
        (gamma, sigma, n_points // 5),
        (sigma, M, n_points // 5),
        (M, K, n_points // 5),
        (K, lambda_point, n_points // 5),
        (lambda_point, gamma, n_points // 5)
    ]
    
    k_path = []
    labels = ['Γ', 'Σ', 'M', 'K', 'Λ', 'Γ']
    positions = [0]
    
    for i, (start, end, n_seg) in enumerate(segments):
        if i == 0:
            t = np.linspace(0, 1, n_seg)
        else:
            t = np.linspace(0, 1, n_seg)[1:]
        
        segment = start[np.newaxis, :] + t[:, np.newaxis] * (end - start)[np.newaxis, :]
        k_path.extend(segment)
        
        if i < len(segments) - 1:
            positions.append(len(k_path))
    
    # Add final position
    positions.append(len(k_path) - 1)
    
    k_path = np.array(k_path)
    k_path_tensor = torch.tensor(k_path, dtype=torch.float32, device=device)
    
    return k_path_tensor, labels, positions


def get_k_points() -> torch.Tensor:
    """Get all 6 K-points where the Dirac cones are located
    
    Following v27/v28 convention using reciprocal lattice vectors.
    K points are at (2/3, 1/3) and equivalent positions in reciprocal space.
    """
    a = LATTICE_CONSTANT
    
    # Reciprocal lattice vectors
    b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
    b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
    
    # The 6 K-points in the first Brillouin zone
    # K points are at corners of hexagonal BZ
    K1 = (2*b1 + b2) / 3    # K at (2/3, 1/3)
    K2 = (b1 + 2*b2) / 3    # K' at (1/3, 2/3)
    K3 = (b1 - b2) / 3      # at (1/3, -1/3)
    K4 = (-b1 - 2*b2) / 3   # at (-1/3, -2/3)
    K5 = (-2*b1 - b2) / 3   # at (-2/3, -1/3)
    K6 = (-b1 + b2) / 3     # at (-1/3, 1/3)
    
    K_points = np.array([K1, K2, K3, K4, K5, K6])
    
    return torch.tensor(K_points, dtype=torch.float32, device=device)


# Unit tests
def test_reciprocal_basis():
    """Test that reciprocal basis satisfies ai·bj = 2π δij"""
    a = LATTICE_CONSTANT
    a1 = np.array([np.sqrt(3)*a/2, a/2])
    a2 = np.array([np.sqrt(3)*a/2, -a/2])
    
    b1, b2 = get_reciprocal_basis()
    
    # Check orthogonality relations
    assert np.abs(np.dot(a1, b1) - 2*np.pi) < 1e-10, "a1·b1 ≠ 2π"
    assert np.abs(np.dot(a1, b2)) < 1e-10, "a1·b2 ≠ 0"
    assert np.abs(np.dot(a2, b1)) < 1e-10, "a2·b1 ≠ 0"
    assert np.abs(np.dot(a2, b2) - 2*np.pi) < 1e-10, "a2·b2 ≠ 2π"
    print("✓ Reciprocal basis test passed")


def test_k_points_zero_gap():
    """Test that all K-points have zero gap"""
    tb_model = TightBindingTarget()
    K_points = get_k_points()
    
    with torch.no_grad():
        tb_output = tb_model(K_points)
        gaps = tb_output['gap'].cpu().numpy()
    
    max_gap = np.max(np.abs(gaps))
    assert max_gap < 1e-5, f"K-points don't have zero gap: max = {max_gap}"
    print("✓ K-points zero gap test passed")


def test_gamma_point_energy():
    """Test energy at Γ point"""
    tb_model = TightBindingTarget()
    gamma_point = torch.zeros(1, 2, device=device)
    
    with torch.no_grad():
        tb_output = tb_model(gamma_point)
    
    expected_energy = 3 * HOPPING_PARAMETER
    actual_energy = tb_output['conduction'][0].item()
    
    assert np.abs(actual_energy - expected_energy) < 1e-3, \
        f"Γ point energy wrong: {actual_energy} vs {expected_energy}"
    print("✓ Gamma point energy test passed")


# Restrict exports
__all__ = ['TightBindingTarget', 'get_reciprocal_basis', 'generate_k_path', 
           'get_k_points', 'LATTICE_CONSTANT', 'HOPPING_PARAMETER', 'device',
           'test_reciprocal_basis', 'test_k_points_zero_gap', 'test_gamma_point_energy']


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests for correct TB implementation...")
    test_reciprocal_basis()
    test_k_points_zero_gap()
    test_gamma_point_energy()
    print("\n✓ All tests passed!")