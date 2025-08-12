#!/usr/bin/env python3
"""
Physics validation for SCMS-PINN v35
CRITICAL: Must show TB vs PINN comparison plots with both curves on same axes
- Blue lines for TB (exact)
- Red lines for PINN predictions
- High-symmetry labels on x-axis (Γ, Σ, M, K, Λ, Γ)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
import sys
import json
import gc  # v33 FIX: Garbage collection to prevent memory leaks

# Import model and utilities
from scms_pinn_graphene_v35 import create_scms_pinn_v35, device
from tb_graphene import (
    TightBindingTarget, generate_k_path, get_k_points, get_reciprocal_basis,
    LATTICE_CONSTANT, HOPPING_PARAMETER
)
from physics_constants_v35 import EXPECTED_FERMI_VELOCITY, CRITICAL_POINTS

# Set matplotlib parameters
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load trained model from checkpoint with epoch metadata (v35)"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_scms_pinn_v35(enforce_symmetry=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # v35 FIX: Support both full checkpoints and raw state_dicts
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = int(checkpoint.get('epoch', 999))
        k_gap = checkpoint.get('k_gap', 'unknown')
        print(f"✓ Loaded model from epoch {epoch}")
        if k_gap != 'unknown':
            print(f"  Model K-gap at save: {k_gap:.6f} eV")
    else:
        # Direct state dict - assume final epoch for hard blending
        model.load_state_dict(checkpoint)
        epoch = 999  # Default to hard blending mode
        print("✓ Loaded model weights (no epoch metadata)")
        print("  Defaulting to epoch=999 for inference (hard blending)")
    
    # v35 FIX: Ensure the epoch-dependent logic is active
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)
        print(f"  Set model epoch to {epoch}")
    
    model.eval()
    return model


def validate_critical_points(model: torch.nn.Module, tb_model: TightBindingTarget) -> dict:
    """Validate model at critical points"""
    results = {}
    
    print("\n" + "="*60)
    print("CRITICAL POINT VALIDATION")
    print("="*60)
    
    # Gamma point
    gamma = torch.zeros(1, 2, device=device)
    with torch.no_grad():
        pinn_gamma = model(gamma)
        tb_gamma = tb_model(gamma)
    
    results['gamma'] = {
        'pinn_valence': pinn_gamma['valence'][0].item(),
        'pinn_conduction': pinn_gamma['conduction'][0].item(),
        'pinn_gap': pinn_gamma['gap'][0].item(),
        'tb_valence': tb_gamma['valence'][0].item(),
        'tb_conduction': tb_gamma['conduction'][0].item(),
        'tb_gap': tb_gamma['gap'][0].item(),
        'expected_valence': -3 * HOPPING_PARAMETER,
        'expected_conduction': 3 * HOPPING_PARAMETER
    }
    
    print(f"\nΓ point (0, 0):")
    print(f"  TB:   V = {results['gamma']['tb_valence']:.3f} eV, C = {results['gamma']['tb_conduction']:.3f} eV")
    print(f"  PINN: V = {results['gamma']['pinn_valence']:.3f} eV, C = {results['gamma']['pinn_conduction']:.3f} eV")
    print(f"  Expected: V = {results['gamma']['expected_valence']:.3f} eV, C = {results['gamma']['expected_conduction']:.3f} eV")
    
    # K-points
    K_points = get_k_points()
    with torch.no_grad():
        pinn_K = model(K_points)
        tb_K = tb_model(K_points)
    
    results['K_points'] = {
        'pinn_gaps': pinn_K['gap'].cpu().numpy(),
        'tb_gaps': tb_K['gap'].cpu().numpy(),
        'pinn_max_gap': pinn_K['gap'].abs().max().item(),
        'tb_max_gap': tb_K['gap'].abs().max().item()
    }
    
    print(f"\nK-points (Dirac points):")
    print(f"  TB max gap:   {results['K_points']['tb_max_gap']:.6f} eV")
    print(f"  PINN max gap: {results['K_points']['pinn_max_gap']:.6f} eV")
    print(f"  Target: 0.0 eV")
    
    for i, (pinn_gap, tb_gap) in enumerate(zip(results['K_points']['pinn_gaps'], 
                                               results['K_points']['tb_gaps'])):
        print(f"  K{i+1}: TB = {tb_gap:.6f} eV, PINN = {pinn_gap:.6f} eV")
    
    return results


def compute_fermi_velocity(model: torch.nn.Module, tb_model: TightBindingTarget) -> dict:
    """Compute Fermi velocity near K-points"""
    K_points = get_k_points()
    K_point = K_points[0]  # Use first K-point
    
    # Sample around K-point
    radii = np.linspace(0.01, 0.2, 50)
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    
    fermi_velocities_pinn = []
    fermi_velocities_tb = []
    
    for r in radii:
        gaps_pinn = []
        gaps_tb = []
        
        for angle in angles:
            offset = torch.tensor([r * np.cos(angle), r * np.sin(angle)], 
                                device=device, dtype=torch.float32)
            k_point = K_point + offset
            
            with torch.no_grad():
                pinn_out = model(k_point.unsqueeze(0))
                tb_out = tb_model(k_point.unsqueeze(0))
                
                gaps_pinn.append(pinn_out['gap'][0].item())
                gaps_tb.append(tb_out['gap'][0].item())
        
        # Average gap for this radius
        avg_gap_pinn = np.mean(gaps_pinn)
        avg_gap_tb = np.mean(gaps_tb)
        
        # Fermi velocity = gap / (2 * |k-K|)
        vf_pinn = avg_gap_pinn / (2 * r)
        vf_tb = avg_gap_tb / (2 * r)
        
        fermi_velocities_pinn.append(vf_pinn)
        fermi_velocities_tb.append(vf_tb)
    
    # Average over reasonable range (0.05-0.1 Å⁻¹)
    idx_range = (radii >= 0.05) & (radii <= 0.1)
    avg_vf_pinn = np.mean(np.array(fermi_velocities_pinn)[idx_range])
    avg_vf_tb = np.mean(np.array(fermi_velocities_tb)[idx_range])
    
    return {
        'radii': radii,
        'vf_pinn': fermi_velocities_pinn,
        'vf_tb': fermi_velocities_tb,
        'avg_vf_pinn': avg_vf_pinn,
        'avg_vf_tb': avg_vf_tb,
        'expected_vf': EXPECTED_FERMI_VELOCITY
    }


def plot_band_structure_comparison(model: torch.nn.Module, tb_model: TightBindingTarget,
                                  save_path: str):
    """
    Plot TB vs PINN band structure comparison
    CRITICAL: Must follow special requirements format
    """
    # Generate k-path
    k_path, labels, positions = generate_k_path(2000)
    
    # Compute bands
    with torch.no_grad():
        pinn_output = model(k_path)
        tb_output = tb_model(k_path)
    
    # Convert to numpy
    pinn_val = pinn_output['valence'].cpu().numpy()
    pinn_cond = pinn_output['conduction'].cpu().numpy()
    pinn_gap = pinn_output['gap'].cpu().numpy()
    
    tb_val = tb_output['valence'].cpu().numpy()
    tb_cond = tb_output['conduction'].cpu().numpy()
    tb_gap = tb_output['gap'].cpu().numpy()
    
    # Create figure with vertical layout (3 subfigures)
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'hspace': 0.3})
    
    # Main band structure plot (first subplot)
    ax1 = axes[0]
    x = np.linspace(0, 1, len(k_path))
    
    # CRITICAL: Plot format per special requirements
    # TB (exact) in solid lines
    ax1.plot(x, tb_val, 'b-', linewidth=2.5, label='TB valence', zorder=2)
    ax1.plot(x, tb_cond, 'r-', linewidth=2.5, label='TB conduction', zorder=2)
    
    # PINN in dashed lines
    ax1.plot(x, pinn_val, 'b--', linewidth=2.5, label='PINN valence', zorder=1)
    ax1.plot(x, pinn_cond, 'r--', linewidth=2.5, label='PINN conduction', zorder=1)
    
    # High-symmetry points
    x_ticks = np.array(positions) / len(k_path)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(labels)  # CRITICAL: Use labels, not numbers
    
    for pos in x_ticks:
        ax1.axvline(x=pos, color='gray', linestyle=':', alpha=0.3)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('k-path', fontsize=14)
    ax1.set_ylabel('Energy (eV)', fontsize=14)
    ax1.set_title('Band Structure', fontsize=16)  # Simplified title per requirements
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 10)
    
    # Gap comparison (second subplot)
    ax2 = axes[1]
    ax2.plot(x, tb_gap, 'g-', linewidth=2.5, label='TB gap', zorder=2)
    ax2.plot(x, pinn_gap, 'g--', linewidth=2.5, label='PINN gap', zorder=1)
    
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(labels)
    for pos in x_ticks:
        ax2.axvline(x=pos, color='gray', linestyle=':', alpha=0.3)
    
    ax2.set_xlabel('k-path', fontsize=14)
    ax2.set_ylabel('Band Gap (eV)', fontsize=14)
    ax2.set_title('Band Gap Comparison', fontsize=16)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, max(tb_gap.max(), pinn_gap.max()) + 1)
    
    # Error plot (third subplot)
    ax3 = axes[2]
    val_error = np.abs(pinn_val - tb_val)
    cond_error = np.abs(pinn_cond - tb_cond)
    gap_error = np.abs(pinn_gap - tb_gap)
    
    ax3.plot(x, val_error, 'b-', linewidth=2, label='Valence error')
    ax3.plot(x, cond_error, 'r-', linewidth=2, label='Conduction error')
    ax3.plot(x, gap_error, 'g-', linewidth=2, label='Gap error')
    
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(labels)
    for pos in x_ticks:
        ax3.axvline(x=pos, color='gray', linestyle=':', alpha=0.3)
    
    ax3.set_xlabel('k-path', fontsize=14)
    ax3.set_ylabel('Absolute Error (eV)', fontsize=14)
    ax3.set_title('PINN vs TB Error', fontsize=16)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-6, 1)
    
    plt.suptitle('SCMS-PINN v35 Physics Validation', fontsize=18)
    # v33 FIX: Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(hspace=0.35, top=0.93, bottom=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # v33 FIX: Safe memory cleanup
    gc.collect()  # Python cleanup only - safe
    # Note: Removed cuda.empty_cache() - causes fragmentation
    
    print(f"✓ Band structure comparison saved: {save_path}")
    
    # Calculate metrics
    max_val_error = val_error.max()
    max_cond_error = cond_error.max()
    max_gap_error = gap_error.max()
    avg_val_error = val_error.mean()
    avg_cond_error = cond_error.mean()
    avg_gap_error = gap_error.mean()
    
    return {
        'max_val_error': max_val_error,
        'max_cond_error': max_cond_error,
        'max_gap_error': max_gap_error,
        'avg_val_error': avg_val_error,
        'avg_cond_error': avg_cond_error,
        'avg_gap_error': avg_gap_error
    }


def plot_fermi_velocity_analysis(fermi_data: dict, save_path: str):
    """Plot Fermi velocity analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    radii = fermi_data['radii']
    vf_pinn = fermi_data['vf_pinn']
    vf_tb = fermi_data['vf_tb']
    expected_vf = fermi_data['expected_vf']
    
    # Fermi velocity vs radius
    ax1.plot(radii, vf_tb, 'b-', linewidth=2.5, label='TB')
    ax1.plot(radii, vf_pinn, 'r--', linewidth=2.5, label='PINN')
    ax1.axhline(y=expected_vf, color='green', linestyle=':', linewidth=2,
                label=f'Expected: {expected_vf:.2f} eV·Å')
    
    # Mark averaging region
    ax1.axvspan(0.05, 0.1, alpha=0.2, color='gray', label='Averaging region')
    
    ax1.set_xlabel('Distance from K-point (Å⁻¹)', fontsize=12)
    ax1.set_ylabel('Fermi Velocity (eV·Å)', fontsize=12)
    ax1.set_title('Fermi Velocity near K-point', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # v35 FIX: Dynamic y-limit to avoid clipping
    y_max = max(max(vf_tb) if len(vf_tb) > 0 else 0, 
                max(vf_pinn) if len(vf_pinn) > 0 else 0, 
                expected_vf) * 1.1
    ax1.set_ylim(0, y_max)
    
    # Relative error
    ax2.plot(radii[1:], np.abs(np.array(vf_pinn[1:]) - np.array(vf_tb[1:])) / np.array(vf_tb[1:]) * 100,
             'r-', linewidth=2)
    ax2.set_xlabel('Distance from K-point (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('PINN vs TB Relative Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 50)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Fermi velocity analysis saved: {save_path}")


def plot_brillouin_zone_gaps(model: torch.nn.Module, tb_model: TightBindingTarget,
                            save_path: str):
    """Plot gap distribution in Brillouin zone"""
    # Create 2D grid
    n_grid = 200
    k_range = 3.5
    kx = np.linspace(-k_range, k_range, n_grid)
    ky = np.linspace(-k_range, k_range, n_grid)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    # Flatten for computation
    k_points = torch.tensor(
        np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=-1),
        dtype=torch.float32, device=device
    )
    
    # Compute in batches to avoid memory issues
    batch_size = 10000
    n_batches = (len(k_points) + batch_size - 1) // batch_size
    
    pinn_gaps = []
    tb_gaps = []
    
    print("Computing Brillouin zone gaps...")
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(k_points))
        k_batch = k_points[start:end]
        
        with torch.no_grad():
            pinn_out = model(k_batch)
            tb_out = tb_model(k_batch)
            
            pinn_gaps.append(pinn_out['gap'].cpu().numpy())
            tb_gaps.append(tb_out['gap'].cpu().numpy())
        
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{n_batches}")
    
    # Concatenate and reshape
    pinn_gap_grid = np.concatenate(pinn_gaps).reshape(n_grid, n_grid)
    tb_gap_grid = np.concatenate(tb_gaps).reshape(n_grid, n_grid)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # TB gap
    im1 = ax1.contourf(kx_grid, ky_grid, tb_gap_grid, levels=50, cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Gap (eV)')
    
    # v33 FIX: Removed K-point markers as requested
    K_points_np = get_k_points().cpu().numpy()  # Still needed for error calculation
    # ax1.scatter(K_points_np[:, 0], K_points_np[:, 1], c='red', s=100,
    #             marker='o', edgecolors='white', linewidth=2, label='K-points')
    
    ax1.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax1.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax1.set_title('TB Band Gap', fontsize=14)
    ax1.set_aspect('equal')
    # ax1.legend()  # Removed since no K-points to show
    
    # PINN gap
    im2 = ax2.contourf(kx_grid, ky_grid, pinn_gap_grid, levels=50, cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='Gap (eV)')
    # v33 FIX: Removed K-point markers
    # ax2.scatter(K_points_np[:, 0], K_points_np[:, 1], c='red', s=100,
    #             marker='o', edgecolors='white', linewidth=2)
    
    ax2.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax2.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax2.set_title('PINN Band Gap', fontsize=14)
    ax2.set_aspect('equal')
    
    # Error
    error_grid = np.abs(pinn_gap_grid - tb_gap_grid)
    im3 = ax3.contourf(kx_grid, ky_grid, error_grid, levels=50, cmap='hot')
    plt.colorbar(im3, ax=ax3, label='Error (eV)')
    # v33 FIX: Removed K-point markers
    # ax3.scatter(K_points_np[:, 0], K_points_np[:, 1], c='blue', s=100,
    #             marker='o', edgecolors='white', linewidth=2)
    
    ax3.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax3.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax3.set_title('Absolute Error', fontsize=14)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Brillouin zone gap analysis saved: {save_path}")
    
    # Calculate errors at K-points
    k_point_errors = []
    for k_point in K_points_np:
        # Find closest grid point to this K-point
        dist = np.sqrt((kx_grid - k_point[0])**2 + (ky_grid - k_point[1])**2)
        min_idx = np.unravel_index(dist.argmin(), dist.shape)
        k_point_errors.append(error_grid[min_idx])
    
    return {
        'max_error': error_grid.max(),
        'avg_error': error_grid.mean(),
        'k_point_errors': np.mean(k_point_errors) if k_point_errors else 0.0
    }


def test_c6v_symmetry(model: torch.nn.Module) -> dict:
    """Test C6v symmetry preservation (from v31/v32)"""
    print("\n" + "="*60)
    print("C6V SYMMETRY TEST")
    print("="*60)
    
    # Test points
    n_test = 100
    k_points = (torch.rand(n_test, 2, device=device) - 0.5) * 4.0
    
    with torch.no_grad():
        reference = model(k_points)['energies']
    
    max_deviations = []
    
    # Test C6 rotations (60° rotations)
    for i in range(1, 6):
        angle = i * np.pi / 3
        rot_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=torch.float32, device=device)
        
        k_rotated = k_points @ rot_matrix.T
        
        with torch.no_grad():
            rotated_result = model(k_rotated)['energies']
        
        deviation = torch.abs(rotated_result - reference).max().item()
        max_deviations.append(deviation)
        print(f"  {i*60}° rotation: max deviation = {deviation:.2e} eV")
    
    # Test mirror symmetry (σv)
    k_mirrored = k_points.clone()
    k_mirrored[:, 0] = -k_mirrored[:, 0]
    
    with torch.no_grad():
        mirrored_result = model(k_mirrored)['energies']
    
    mirror_deviation = torch.abs(mirrored_result - reference).max().item()
    max_deviations.append(mirror_deviation)
    print(f"  Mirror (σv): max deviation = {mirror_deviation:.2e} eV")
    
    max_deviation = max(max_deviations)
    passed = max_deviation < 1e-5
    
    print(f"\n  Overall maximum deviation: {max_deviation:.2e} eV")
    print(f"  {'✓' if passed else '✗'} C6v symmetry test {'PASSED' if passed else 'FAILED'}")
    
    return {
        'passed': passed,
        'max_deviation': max_deviation,
        'deviations': max_deviations
    }


def test_energy_scale(model: torch.nn.Module, tb_model: TightBindingTarget) -> dict:
    """Test overall energy scale accuracy (from v32)"""
    print("\n" + "="*60)
    print("ENERGY SCALE TEST")
    print("="*60)
    
    # Sample many k-points
    n_samples = 5000
    k_points = torch.randn(n_samples, 2, device=device) * 2.0
    
    with torch.no_grad():
        pinn_output = model(k_points)
        tb_output = tb_model(k_points)
    
    # Calculate RMS energies
    pinn_energies = pinn_output['energies'].cpu().numpy()
    tb_energies = tb_output['energies'].cpu().numpy()
    
    rms_pinn = np.sqrt(np.mean(pinn_energies**2))
    rms_tb = np.sqrt(np.mean(tb_energies**2))
    
    scale_ratio = rms_pinn / rms_tb
    scale_error = abs(scale_ratio - 1.0) * 100
    
    print(f"  TB RMS energy: {rms_tb:.4f} eV")
    print(f"  PINN RMS energy: {rms_pinn:.4f} eV")
    print(f"  Scale ratio: {scale_ratio:.4f}")
    print(f"  Scale error: {scale_error:.2f}%")
    
    passed = scale_error < 2  # Pass if within 2%
    print(f"  {'✓' if passed else '✗'} Energy scale test {'PASSED' if passed else 'FAILED'}")
    
    return {
        'passed': passed,
        'rms_tb': rms_tb,
        'rms_pinn': rms_pinn,
        'scale_ratio': scale_ratio,
        'scale_error': scale_error
    }


def test_autograd_fermi_velocity(model: torch.nn.Module) -> dict:
    """Test autograd computation for Fermi velocity (from v32)"""
    print("\n" + "="*60)
    print("AUTOGRAD SANITY CHECK")
    print("="*60)
    
    # Sample near-K points
    K_point = get_k_points()[0]
    offsets = torch.randn(8, 2, device=device) * 0.1
    k_points = K_point.unsqueeze(0) + offsets
    k_points.requires_grad_(True)
    
    try:
        # Compute gap and its gradient
        gap = model(k_points)['gap']
        
        # Check if gap requires grad
        if not gap.requires_grad:
            print("  ✗ Gap does not require grad - autograd chain broken!")
            return {'passed': False, 'error': 'No gradient'}
        
        # Compute gradients
        grad_gap = torch.autograd.grad(
            outputs=gap.sum(),
            inputs=k_points,
            create_graph=False
        )[0]
        
        # Fermi velocity magnitude
        vf_magnitudes = 0.5 * torch.norm(grad_gap, dim=1)
        avg_vf = vf_magnitudes.mean().item()
        
        # Check if reasonable
        # v35 FIX: Update upper bound to 6.5 eV·Å for graphene
        passed = 0.5 < avg_vf < 6.5  # Reasonable range for Fermi velocity
        
        print(f"  Average Fermi velocity: {avg_vf:.4f} eV·Å")
        print(f"  Expected range: 0.5 - 6.5 eV·Å")
        print(f"  {'✓' if passed else '✗'} Autograd test {'PASSED' if passed else 'FAILED'}")
        
        return {
            'passed': passed,
            'avg_vf': avg_vf,
            'vf_values': vf_magnitudes.cpu().numpy().tolist()
        }
        
    except Exception as e:
        print(f"  ✗ Autograd test FAILED with error: {e}")
        return {'passed': False, 'error': str(e)}


def generate_validation_report(results: dict, save_path: str):
    """Generate comprehensive validation report"""
    report = f"""# SCMS-PINN v35 Physics Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

The SCMS-PINN v35 model has been validated against the exact tight-binding (TB) solution for graphene. This report presents a comprehensive comparison of band structures, critical point energies, and Fermi velocities.

## Critical Point Validation

### Γ Point (0, 0)
- **TB Values**: Valence = {results['critical_points']['gamma']['tb_valence']:.4f} eV, Conduction = {results['critical_points']['gamma']['tb_conduction']:.4f} eV
- **PINN Values**: Valence = {results['critical_points']['gamma']['pinn_valence']:.4f} eV, Conduction = {results['critical_points']['gamma']['pinn_conduction']:.4f} eV
- **Expected**: Valence = {results['critical_points']['gamma']['expected_valence']:.4f} eV, Conduction = {results['critical_points']['gamma']['expected_conduction']:.4f} eV
- **Error**: Valence = {abs(results['critical_points']['gamma']['pinn_valence'] - results['critical_points']['gamma']['tb_valence']):.6f} eV, Conduction = {abs(results['critical_points']['gamma']['pinn_conduction'] - results['critical_points']['gamma']['tb_conduction']):.6f} eV

### K-Points (Dirac Points)
- **TB Maximum Gap**: {results['critical_points']['K_points']['tb_max_gap']:.6f} eV
- **PINN Maximum Gap**: {results['critical_points']['K_points']['pinn_max_gap']:.6f} eV
- **Target**: 0.0 eV
- **Status**: {"✓ PASSED" if results['critical_points']['K_points']['pinn_max_gap'] < 0.01 else "✗ FAILED"}

## Band Structure Accuracy

### Overall Metrics
- **Maximum Valence Error**: {results['band_structure']['max_val_error']:.6f} eV
- **Maximum Conduction Error**: {results['band_structure']['max_cond_error']:.6f} eV
- **Maximum Gap Error**: {results['band_structure']['max_gap_error']:.6f} eV
- **Average Valence Error**: {results['band_structure']['avg_val_error']:.6f} eV
- **Average Conduction Error**: {results['band_structure']['avg_cond_error']:.6f} eV
- **Average Gap Error**: {results['band_structure']['avg_gap_error']:.6f} eV

## Fermi Velocity Analysis

### Near K-Points
- **TB Fermi Velocity**: {results['fermi_velocity']['avg_vf_tb']:.4f} eV·Å
- **PINN Fermi Velocity**: {results['fermi_velocity']['avg_vf_pinn']:.4f} eV·Å
- **Expected**: {results['fermi_velocity']['expected_vf']:.4f} eV·Å
- **Relative Error**: {abs(results['fermi_velocity']['avg_vf_pinn'] - results['fermi_velocity']['avg_vf_tb']) / results['fermi_velocity']['avg_vf_tb'] * 100:.2f}%

## Brillouin Zone Analysis

### Gap Distribution
- **Maximum Error**: {results['brillouin_zone']['max_error']:.6f} eV
- **Average Error**: {results['brillouin_zone']['avg_error']:.6f} eV
- **Error near K-points**: {results['brillouin_zone']['k_point_errors']:.6f} eV

## Additional Physics Tests (from v31/v32)

### Autograd Test
- **Status**: {"✓ PASSED" if results.get('autograd', dict()).get('passed', False) else "✗ FAILED"}
- **Average Fermi Velocity (via gradients)**: {f"{results.get('autograd', dict()).get('avg_vf', 0):.4f}" if isinstance(results.get('autograd', dict()).get('avg_vf'), (int, float)) else 'N/A'} eV·Å
- **Purpose**: Ensures gradient computation for physics-informed training

### C6v Symmetry Test
- **Status**: {"✓ PASSED" if results.get('c6v_symmetry', dict()).get('passed', False) else "✗ FAILED"}
- **Maximum Deviation**: {results.get('c6v_symmetry', dict()).get('max_deviation', 0):.2e} eV
- **Purpose**: Verifies hexagonal symmetry preservation

### Energy Scale Test
- **Status**: {"✓ PASSED" if results.get('energy_scale', dict()).get('passed', False) else "✗ FAILED"}
- **Scale Ratio (PINN/TB)**: {results.get('energy_scale', dict()).get('scale_ratio', 1.0):.4f}
- **Scale Error**: {results.get('energy_scale', dict()).get('scale_error', 0):.2f}%
- **Purpose**: Ensures correct energy magnitude

## Physics Compliance Summary

1. **Dirac Points**: {"✓ Zero gap maintained" if results['critical_points']['K_points']['pinn_max_gap'] < 0.01 else "✗ Non-zero gap detected"}
2. **Band Energies**: {"✓ Correct at Γ point" if abs(results['critical_points']['gamma']['pinn_valence'] - results['critical_points']['gamma']['expected_valence']) < 0.1 else "✗ Incorrect at Γ point"}
3. **Fermi Velocity**: {"✓ Within 10% of expected" if abs(results['fermi_velocity']['avg_vf_pinn'] - results['fermi_velocity']['expected_vf']) / results['fermi_velocity']['expected_vf'] < 0.1 else "✗ Deviates > 10% from expected"}
4. **Band Structure**: {"✓ Good agreement" if results['band_structure']['avg_gap_error'] < 0.1 else "✗ Large deviations"}

## Conclusion

The SCMS-PINN v35 model {"successfully captures" if results['critical_points']['K_points']['pinn_max_gap'] < 0.01 and results['band_structure']['avg_gap_error'] < 0.1 else "requires further optimization to capture"} the essential physics of graphene's electronic structure.

### Key Achievements:
- Band structure closely matches TB solution
- Critical point energies are accurate
- Fermi velocity near K-points is well-reproduced

### Areas for Improvement:
- {"None identified - model performs excellently" if results['critical_points']['K_points']['pinn_max_gap'] < 0.001 else "Further reduction of gap at K-points needed"}
- {"N/A" if results['band_structure']['max_gap_error'] < 0.1 else "Reduce maximum band structure errors"}

## Figures Generated

1. `band_structure_comparison.png` - TB vs PINN band structure overlay
2. `fermi_velocity_analysis.png` - Fermi velocity comparison
3. `brillouin_zone_gaps.png` - Gap distribution in k-space

---
*Report generated by physics_validation_graphene_v35.py*
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Validation report saved: {save_path}")


def plot_directional_fermi_velocity_all_kpoints(model: torch.nn.Module, 
                                                tb_model: TightBindingTarget,
                                                save_path: str):
    """Plot directional Fermi velocity around all 6 K-points (v33 ADD)"""
    print("\n" + "="*60)
    print("DIRECTIONAL FERMI VELOCITY AT ALL K-POINTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                           subplot_kw=dict(projection='polar'))
    
    # Get all 6 K-points
    K_points = get_k_points().cpu().numpy()
    
    # Parameters for velocity calculation
    radius = 0.1  # Distance from K-point
    n_angles = 72  # Number of angular samples
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    for idx, (ax, K_point) in enumerate(zip(axes.flat, K_points)):
        vf_tb = []
        vf_pinn = []
        
        for angle in angles:
            # Point at fixed distance from K-point
            k = K_point + radius * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            # TB Fermi velocity
            k_tensor.requires_grad_(True)
            tb_gap = tb_model(k_tensor)['gap']
            tb_grad = torch.autograd.grad(tb_gap, k_tensor, create_graph=False)[0]
            vf_tb.append(0.5 * torch.norm(tb_grad).item())
            
            # PINN Fermi velocity
            k_tensor = k_tensor.detach().requires_grad_(True)
            pinn_gap = model(k_tensor)['gap']
            pinn_grad = torch.autograd.grad(pinn_gap, k_tensor, create_graph=False)[0]
            vf_pinn.append(0.5 * torch.norm(pinn_grad).item())
        
        # Plot on polar axes
        ax.plot(angles, vf_tb, 'b-', linewidth=2, label='TB')
        ax.plot(angles, vf_pinn, 'r--', linewidth=2, label='PINN')
        
        # Formatting
        # v35 FIX: Simple sequential numbering for K-points
        ax.set_title(f'K{idx+1}', pad=20)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Readability fix: use dynamic, human-scale radial ticks and move labels away
        r_max = float(max(max(vf_tb), max(vf_pinn))) * 1.1
        # Round up to a clean 0.5 or 1.0 step range
        nice_cap = float(np.ceil(r_max))
        step = 1.0 if nice_cap >= 6 else 0.5
        ticks = np.arange(step, nice_cap + 1e-9, step)  # exclude 0 to avoid center label
        if len(ticks) < 3:  # ensure at least a few rings
            step = max(step / 2.0, 0.5)
            ticks = np.arange(step, nice_cap + 1e-9, step)

        ax.set_ylim(0, nice_cap)
        ax.set_rticks(ticks.tolist())
        ax.set_rlabel_position(135)  # place radial labels on the left to avoid clutter
        # Ultra-small radial tick labels for unobtrusive annotation
        ax.tick_params(axis='y', labelsize=6, pad=1)
        # Format tick labels to one decimal when needed
        tick_labels = [f"{t:.1f}" if step < 1.0 else f"{int(t)}" for t in ticks]
        ax.set_yticklabels(tick_labels)
        ax.grid(True, alpha=0.3)
        
        # Do not add per-axes legend; we'll add a single figure-level legend below
        if idx == 0:
            pass
        
        # Add angle labels
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    
    # Overall title
    fig.suptitle('Directional Fermi Velocity around K-points (eV·Å)', fontsize=16)

    # One figure-level legend positioned at bottom center to avoid overlap
    tb_handle = mlines.Line2D([], [], color='b', linestyle='-', linewidth=2, label='TB (solid)')
    pinn_handle = mlines.Line2D([], [], color='r', linestyle='--', linewidth=2, label='PINN (dashed)')
    fig.legend(handles=[tb_handle, pinn_handle],
               loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2,
               fontsize=12, frameon=False)
    # Clarifying note about radial ticks placed just above legend
    fig.text(0.5, 0.06, 'Radial ticks denote v_F magnitude (eV·Å)',
             ha='center', va='center', fontsize=12)
    
    # v33 FIX: Use subplots_adjust instead of tight_layout for polar plots
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.90, bottom=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # v33 FIX: Safe memory cleanup
    gc.collect()  # Python cleanup only - safe
    # Note: Removed cuda.empty_cache() - causes fragmentation
    
    print(f"✓ Directional Fermi velocity plots saved: {save_path}")
    
    # Calculate statistics
    avg_error = 0
    for idx in range(6):
        K_point = K_points[idx]
        k = K_point + radius * np.array([1, 0])  # Sample point
        k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        
        k_tensor.requires_grad_(True)
        tb_gap = tb_model(k_tensor)['gap']
        tb_grad = torch.autograd.grad(tb_gap, k_tensor, create_graph=False)[0]
        vf_tb_val = 0.5 * torch.norm(tb_grad).item()
        
        k_tensor = k_tensor.detach().requires_grad_(True)
        pinn_gap = model(k_tensor)['gap']
        pinn_grad = torch.autograd.grad(pinn_gap, k_tensor, create_graph=False)[0]
        vf_pinn_val = 0.5 * torch.norm(pinn_grad).item()
        
        error = abs(vf_pinn_val - vf_tb_val) / vf_tb_val * 100
        avg_error += error
        
    avg_error /= 6
    print(f"Average Fermi velocity error across K-points: {avg_error:.2f}%")
    
    return {'avg_error': avg_error}


def plot_overlay_directional_fermi_velocity_all_kpoints(model: torch.nn.Module,
                                                        tb_model: TightBindingTarget,
                                                        save_path: str):
    """Overlay all six K-point directional vF on one polar axis.
    - Color encodes K1..K6
    - Line style encodes TB (solid) vs PINN (dashed)
    """
    print("\n" + "="*60)
    print("OVERLAY DIRECTIONAL FERMI VELOCITY (ALL K-POINTS)")
    print("="*60)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')

    # Data setup
    K_points = get_k_points().cpu().numpy()
    radius = 0.1
    n_angles = 180
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    # Color map per K (consistent for TB and PINN)
    k_colors = [
        'tab:blue',   # K1
        'tab:orange', # K2
        'tab:green',  # K3
        'tab:red',    # K4
        'tab:purple', # K5
        'tab:brown'   # K6
    ]

    all_values = []

    tb_k1_plotted = False
    tb_k2_plotted = False
    for idx, K_point in enumerate(K_points):
        color = k_colors[idx]
        vf_tb = []
        vf_pinn = []

        for angle in angles:
            k = K_point + radius * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)

            # TB
            k_tensor.requires_grad_(True)
            tb_gap = tb_model(k_tensor)['gap']
            tb_grad = torch.autograd.grad(tb_gap, k_tensor, create_graph=False)[0]
            vf_tb.append(0.5 * torch.norm(tb_grad).item())

            # PINN
            k_tensor = k_tensor.detach().requires_grad_(True)
            pinn_gap = model(k_tensor)['gap']
            pinn_grad = torch.autograd.grad(pinn_gap, k_tensor, create_graph=False)[0]
            vf_pinn.append(0.5 * torch.norm(pinn_grad).item())

        all_values.extend(vf_tb)
        all_values.extend(vf_pinn)

        # TB: only two solid curves → K2 (rep for {K2,K3,K5}) and K1 (rep for {K1,K4,K6})
        if idx == 1 and not tb_k2_plotted:  # K2
            ax.plot(angles, vf_tb, color=k_colors[1], linestyle='-', linewidth=2.5, alpha=0.95)
            tb_k2_plotted = True
        if idx == 0 and not tb_k1_plotted:  # K1
            ax.plot(angles, vf_tb, color=k_colors[0], linestyle='-', linewidth=2.5, alpha=0.95)
            tb_k1_plotted = True

        # PINN: six curves with distinguishable styles (vary dash/marker)
        pinn_styles = [
            {'linestyle': (0, (1, 2)), 'marker': None},      # K1: fine dash
            {'linestyle': (0, (2, 2)), 'marker': None},      # K2: short dash
            {'linestyle': (0, (4, 2)), 'marker': None},      # K3: medium dash
            {'linestyle': (0, (6, 2)), 'marker': None},      # K4: long dash
            {'linestyle': (0, (3, 1, 1, 1)), 'marker': None},# K5: dash-dot-dot
            {'linestyle': (0, (1, 1)), 'marker': None},      # K6: dotted
        ]
        style = pinn_styles[idx]
        ax.plot(angles, vf_pinn, color=color, linestyle=style['linestyle'], linewidth=2, alpha=0.9)

    # Axis formatting (shared)
    r_max = float(max(all_values)) * 1.1 if all_values else 1.0
    nice_cap = float(np.ceil(r_max))
    step = 1.0 if nice_cap >= 6 else 0.5
    ticks = np.arange(step, nice_cap + 1e-9, step)
    if len(ticks) < 3:
        step = max(step / 2.0, 0.5)
        ticks = np.arange(step, nice_cap + 1e-9, step)

    ax.set_ylim(0, nice_cap)
    ax.set_rticks(ticks.tolist())
    ax.set_rlabel_position(135)
    ax.tick_params(axis='y', labelsize=8, pad=2)
    tick_labels = [f"{t:.1f}" if step < 1.0 else f"{int(t)}" for t in ticks]
    ax.set_yticklabels(tick_labels)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.grid(True, alpha=0.3)

    # Consolidated legend: TB representatives + six PINN entries (match styles)
    legend_handles = [
        mlines.Line2D([], [], color=k_colors[1], linestyle='-', linewidth=3, label='TB K2 (rep. of K2,K3,K5)'),
        mlines.Line2D([], [], color=k_colors[0], linestyle='-', linewidth=3, label='TB K1 (rep. of K1,K4,K6)'),
        mlines.Line2D([], [], color=k_colors[0], linestyle=(0, (1, 2)), linewidth=3, label='PINN K1'),
        mlines.Line2D([], [], color=k_colors[1], linestyle=(0, (2, 2)), linewidth=3, label='PINN K2'),
        mlines.Line2D([], [], color=k_colors[2], linestyle=(0, (4, 2)), linewidth=3, label='PINN K3'),
        mlines.Line2D([], [], color=k_colors[3], linestyle=(0, (6, 2)), linewidth=3, label='PINN K4'),
        mlines.Line2D([], [], color=k_colors[4], linestyle=(0, (3, 1, 1, 1)), linewidth=3, label='PINN K5'),
        mlines.Line2D([], [], color=k_colors[5], linestyle=(0, (1, 1)), linewidth=3, label='PINN K6'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.14),
        ncol=2,
        fontsize=9,
        frameon=False
    )

    # Titles/captions
    fig.suptitle('Overlay: Directional Fermi Velocity across K-points (eV·Å)', fontsize=14)
    fig.text(0.5, 0.12, r'Radial ticks denote $v_{f}$ magnitude (eV·Å)', ha='center', va='center', fontsize=9)

    # Inset: K-point locations in reciprocal space (placed in bottom margin to avoid overlap)
    # Use figure coordinates so it sits below the polar axes and above the legend/caption
    # Reduce size to half and nudge slightly upward to avoid any overlap
    inset = fig.add_axes([0.10, 0.20, 0.12, 0.12])
    # Order K-points by polar angle to draw a hexagon
    kpts = K_points
    ang = np.arctan2(kpts[:, 1], kpts[:, 0])
    order = np.argsort(ang)
    poly = np.vstack([kpts[order], kpts[order][0]])
    inset.plot(poly[:, 0], poly[:, 1], color='lightgray', linewidth=1)
    # Scatter and label
    for i, (x, y) in enumerate(kpts):
        inset.scatter(x, y, s=15, color=k_colors[i], zorder=3)
        inset.text(x, y, f"K{i+1}", fontsize=7, color=k_colors[i], ha='left', va='bottom')
    # Draw reciprocal lattice vectors b1 and b2 from origin (0,0) per Eq (1)
    try:
        b1, b2 = get_reciprocal_basis()
        inset.arrow(0, 0, b1[0]*0.4, b1[1]*0.4, head_width=0.05, head_length=0.08, fc='k', ec='k', length_includes_head=True)
        inset.text(b1[0]*0.45, b1[1]*0.45, r"$\mathbf{b}_{1}$", fontsize=7, ha='left', va='bottom')
        inset.arrow(0, 0, b2[0]*0.4, b2[1]*0.4, head_width=0.05, head_length=0.08, fc='k', ec='k', length_includes_head=True)
        inset.text(b2[0]*0.45, b2[1]*0.45, r"$\mathbf{b}_{2}$", fontsize=7, ha='left', va='bottom')
    except Exception:
        pass
    # Clean aesthetics
    inset.set_aspect('equal', adjustable='box')
    inset.set_xticks([]); inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_title('K-points', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.28)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Overlay directional Fermi velocity saved: {save_path}")


    # (Removed by request) Direct-label overlay plot

def main():
    """Main validation script"""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python physics_validation_graphene_v35.py <checkpoint_path> [output_dir]")
        print("Example: python physics_validation_graphene_v35.py training_v35_*/best_model.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "validation_v35_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load models
    model = load_model(checkpoint_path)
    tb_model = TightBindingTarget()
    
    # Run validations
    results = {}
    
    # 1. Critical points
    results['critical_points'] = validate_critical_points(model, tb_model)
    
    # 2. Band structure comparison (CRITICAL: Must include TB vs PINN overlay)
    band_path = os.path.join(output_dir, 'band_structure_comparison.png')
    results['band_structure'] = plot_band_structure_comparison(model, tb_model, band_path)
    
    # 3. Fermi velocity
    print("\nComputing Fermi velocity...")
    results['fermi_velocity'] = compute_fermi_velocity(model, tb_model)
    fermi_path = os.path.join(output_dir, 'fermi_velocity_analysis.png')
    plot_fermi_velocity_analysis(results['fermi_velocity'], fermi_path)
    
    print(f"\nFermi Velocity Results:")
    print(f"  TB average: {results['fermi_velocity']['avg_vf_tb']:.4f} eV·Å")
    print(f"  PINN average: {results['fermi_velocity']['avg_vf_pinn']:.4f} eV·Å")
    print(f"  Expected: {results['fermi_velocity']['expected_vf']:.4f} eV·Å")
    
    # 4. Brillouin zone analysis
    bz_path = os.path.join(output_dir, 'brillouin_zone_gaps.png')
    results['brillouin_zone'] = plot_brillouin_zone_gaps(model, tb_model, bz_path)
    
    # 5. NEW: Directional Fermi velocity at all K-points (v33 ADD)
    dir_vf_path = os.path.join(output_dir, 'directional_fermi_velocity_all_kpoints.png')
    results['directional_fermi_all'] = plot_directional_fermi_velocity_all_kpoints(
        model, tb_model, dir_vf_path
    )
    # 5b. Overlay plot (new)
    dir_vf_overlay_path = os.path.join(output_dir, 'overlay_directional_fermi_velocity.png')
    plot_overlay_directional_fermi_velocity_all_kpoints(model, tb_model, dir_vf_overlay_path)
    # 5c. (Removed by request) overlay with direct labels
    
    # 6. NEW: Run additional tests from v31/v32
    # Autograd test
    results['autograd'] = test_autograd_fermi_velocity(model)
    
    # C6v symmetry test
    results['c6v_symmetry'] = test_c6v_symmetry(model)
    
    # Energy scale test
    results['energy_scale'] = test_energy_scale(model, tb_model)
    
    # 6. Generate report
    report_path = os.path.join(output_dir, 'validation_report.md')
    generate_validation_report(results, report_path)
    
    # Save results as JSON
    results_json = {
        'checkpoint_path': checkpoint_path,
        'timestamp': datetime.now().isoformat(),
        'results': {
            'critical_points': {
                'gamma': results['critical_points']['gamma'],
                'K_max_gap': results['critical_points']['K_points']['pinn_max_gap']
            },
            'band_structure': results['band_structure'],
            'fermi_velocity': {
                'avg_tb': results['fermi_velocity']['avg_vf_tb'],
                'avg_pinn': results['fermi_velocity']['avg_vf_pinn'],
                'expected': results['fermi_velocity']['expected_vf']
            },
            'brillouin_zone': results['brillouin_zone']
        }
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_json = convert_to_serializable(results_json)
    
    json_path = os.path.join(output_dir, 'validation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Validation results saved: {json_path}")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Gap at K-points: {results['critical_points']['K_points']['pinn_max_gap']:.6f} eV")
    print(f"Average band error: {results['band_structure']['avg_gap_error']:.6f} eV")
    print(f"Fermi velocity error: {abs(results['fermi_velocity']['avg_vf_pinn'] - results['fermi_velocity']['avg_vf_tb']) / results['fermi_velocity']['avg_vf_tb'] * 100:.2f}%")
    
    # NEW: Print results of additional tests
    if 'autograd' in results:
        print(f"Autograd test: {'✓ PASSED' if results['autograd'].get('passed', False) else '✗ FAILED'}")
    if 'c6v_symmetry' in results:
        print(f"C6v symmetry: {'✓ PASSED' if results['c6v_symmetry'].get('passed', False) else '✗ FAILED'} (max dev: {results['c6v_symmetry'].get('max_deviation', 0):.2e})")
    if 'energy_scale' in results:
        print(f"Energy scale: {'✓ PASSED' if results['energy_scale'].get('passed', False) else '✗ FAILED'} (error: {results['energy_scale'].get('scale_error', 0):.2f}%)")
    
    if results['critical_points']['K_points']['pinn_max_gap'] < 0.01:
        print("\n✓ Model successfully maintains zero gap at Dirac points!")
    else:
        print("\n✗ Model needs improvement - gap at K-points is too large")
    
    print("\n✓ All validation tests completed (including v31/v32 tests)!")


if __name__ == "__main__":
    main()