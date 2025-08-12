#!/usr/bin/env python3
"""
Comprehensive visualization suite for SCMS-PINN v35
Adds missing plots and improves existing visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as patches

# Import model components
from scms_pinn_graphene_v35 import create_scms_pinn_v35, device
from physics_validation_graphene_v35 import load_model
from tb_graphene import TightBindingTarget, get_k_points
from physics_constants_v35 import (
    LATTICE_CONSTANT, HOPPING_PARAMETER, EXPECTED_FERMI_VELOCITY
)

# Create constants dict for compatibility
GRAPHENE_CONSTANTS = {
    'lattice_constant': LATTICE_CONSTANT,
    'hopping_parameter': HOPPING_PARAMETER,
    'expected_fermi_velocity': EXPECTED_FERMI_VELOCITY
}

def get_band_path(n_points: int = 300):
    """Generate k-path for band structure plot"""
    a = LATTICE_CONSTANT
    
    # High symmetry points
    Gamma = [0, 0]
    K = [4*np.pi/(3*a), 0]
    M = [np.pi/a, np.pi/(np.sqrt(3)*a)]
    
    # Create path: Γ -> K -> M -> Γ
    path_segments = []
    labels = ['Γ', 'K', 'M', 'Γ']
    
    # Γ to K
    t = np.linspace(0, 1, n_points//3)
    for ti in t:
        path_segments.append((1-ti)*np.array(Gamma) + ti*np.array(K))
    
    # K to M
    t = np.linspace(0, 1, n_points//3)
    for ti in t[1:]:
        path_segments.append((1-ti)*np.array(K) + ti*np.array(M))
    
    # M to Γ
    t = np.linspace(0, 1, n_points//3)
    for ti in t[1:]:
        path_segments.append((1-ti)*np.array(M) + ti*np.array(Gamma))
    
    k_path = np.array(path_segments)
    
    # Positions for vertical lines
    positions = [0, n_points//3, 2*n_points//3, len(k_path)-1]
    
    return k_path, labels, positions


def plot_band_structure_vertical(model: torch.nn.Module, tb_model: TightBindingTarget,
                                save_path: str):
    """Plot band structure comparison with vertical layout"""
    # Get band path
    k_path, labels, positions = get_band_path(n_points=300)
    k_path_tensor = torch.tensor(k_path, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pinn_output = model(k_path_tensor)
        tb_output = tb_model(k_path_tensor)
    
    # Convert to numpy
    pinn_np = {k: v.cpu().numpy() for k, v in pinn_output.items()}
    tb_np = {k: v.cpu().numpy() for k, v in tb_output.items()}
    
    # Create figure with vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Band structure
    ax = axes[0]
    ax.plot(tb_np['valence'], 'b-', label='TB valence', linewidth=2)
    ax.plot(tb_np['conduction'], 'r-', label='TB conduction', linewidth=2)
    ax.plot(pinn_np['valence'], 'b--', label='PINN valence', linewidth=1.5, alpha=0.7)
    ax.plot(pinn_np['conduction'], 'r--', label='PINN conduction', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    for pos in positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Band Structure', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 10)
    
    # Band gap comparison
    ax = axes[1]
    ax.plot(tb_np['gap'], 'g-', label='TB gap', linewidth=2)
    ax.plot(pinn_np['gap'], 'g--', label='PINN gap', linewidth=1.5, alpha=0.7)
    for pos in positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Band Gap (eV)', fontsize=12)
    ax.set_title('Band Gap Comparison', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Error plot
    ax = axes[2]
    valence_error = np.abs(pinn_np['valence'] - tb_np['valence'])
    conduction_error = np.abs(pinn_np['conduction'] - tb_np['conduction'])
    gap_error = np.abs(pinn_np['gap'] - tb_np['gap'])
    
    ax.semilogy(valence_error, 'b-', label='Valence error', linewidth=1.5)
    ax.semilogy(conduction_error, 'r-', label='Conduction error', linewidth=1.5)
    ax.semilogy(gap_error, 'g-', label='Gap error', linewidth=1.5)
    for pos in positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k-path', fontsize=12)
    ax.set_ylabel('Absolute Error (eV)', fontsize=12)
    ax.set_title('PINN vs TB Error', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-6, 1e0)
    
    # Set x-ticks
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels)
    
    plt.suptitle('SCMS-PINN v35 Physics Validation', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Vertical band structure saved: {save_path}")


def plot_brillouin_zone_clean(model: torch.nn.Module, tb_model: TightBindingTarget,
                             save_path: str):
    """Plot Brillouin zone gaps without artificial points"""
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
    
    # Compute in batches
    batch_size = 10000
    n_batches = (len(k_points) + batch_size - 1) // batch_size
    
    pinn_gaps = []
    tb_gaps = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(k_points))
        k_batch = k_points[start:end]
        
        with torch.no_grad():
            pinn_out = model(k_batch)
            tb_out = tb_model(k_batch)
            
            pinn_gaps.append(pinn_out['gap'].cpu().numpy())
            tb_gaps.append(tb_out['gap'].cpu().numpy())
    
    # Reshape
    pinn_gap_grid = np.concatenate(pinn_gaps).reshape(n_grid, n_grid)
    tb_gap_grid = np.concatenate(tb_gaps).reshape(n_grid, n_grid)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # TB gap
    im1 = ax1.contourf(kx_grid, ky_grid, tb_gap_grid, levels=50, cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Gap (eV)')
    
    # Mark K-points with circles only
    K_points_np = get_k_points().cpu().numpy()
    ax1.scatter(K_points_np[:, 0], K_points_np[:, 1], c='red', s=100,
                marker='o', edgecolors='white', linewidth=2, label='K-points', zorder=5)
    
    ax1.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax1.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax1.set_title('TB Band Gap', fontsize=14)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # PINN gap
    im2 = ax2.contourf(kx_grid, ky_grid, pinn_gap_grid, levels=50, cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='Gap (eV)')
    ax2.scatter(K_points_np[:, 0], K_points_np[:, 1], c='red', s=100,
                marker='o', edgecolors='white', linewidth=2, zorder=5)
    
    ax2.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax2.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax2.set_title('PINN Band Gap', fontsize=14)
    ax2.set_aspect('equal')
    
    # Error
    error_grid = np.abs(pinn_gap_grid - tb_gap_grid)
    im3 = ax3.contourf(kx_grid, ky_grid, error_grid, levels=50, cmap='hot')
    plt.colorbar(im3, ax=ax3, label='Error (eV)')
    ax3.scatter(K_points_np[:, 0], K_points_np[:, 1], c='blue', s=100,
                marker='o', edgecolors='white', linewidth=2, zorder=5)
    
    ax3.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax3.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax3.set_title('Absolute Error', fontsize=14)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Clean Brillouin zone plot saved: {save_path}")


def plot_fermi_velocity_vertical(model: torch.nn.Module, tb_model: TightBindingTarget,
                                save_path: str):
    """Plot Fermi velocity analysis with vertical layout"""
    # Sample points near K-point
    K_point = get_k_points()[0].cpu().numpy()
    distances = np.linspace(0.01, 0.2, 50)
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    
    vf_tb_list = []
    vf_pinn_list = []
    
    for dist in distances:
        vf_tb_angles = []
        vf_pinn_angles = []
        
        for angle in angles:
            k = K_point + dist * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            # TB Fermi velocity
            k_tensor.requires_grad_(True)
            tb_out = tb_model(k_tensor)
            tb_gap = tb_out['gap']
            tb_grad = torch.autograd.grad(tb_gap, k_tensor, create_graph=False)[0]
            vf_tb = 0.5 * torch.norm(tb_grad).item()
            
            # PINN Fermi velocity
            k_tensor = k_tensor.detach()
            k_tensor.requires_grad_(True)
            pinn_out = model(k_tensor)
            pinn_gap = pinn_out['gap']
            pinn_grad = torch.autograd.grad(pinn_gap, k_tensor, create_graph=False)[0]
            vf_pinn = 0.5 * torch.norm(pinn_grad).item()
            
            vf_tb_angles.append(vf_tb)
            vf_pinn_angles.append(vf_pinn)
        
        vf_tb_list.append(np.mean(vf_tb_angles))
        vf_pinn_list.append(np.mean(vf_pinn_angles))
    
    # Create vertical layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Fermi velocity magnitude
    ax = axes[0]
    ax.plot(distances, vf_tb_list, 'b-', linewidth=2, label='TB')
    ax.plot(distances, vf_pinn_list, 'r--', linewidth=2, label='PINN')
    ax.axhline(y=GRAPHENE_CONSTANTS['expected_fermi_velocity'], 
               color='green', linestyle=':', linewidth=2, 
               label=f'Expected: {GRAPHENE_CONSTANTS["expected_fermi_velocity"]:.2f} eV·Å')
    
    # Shade averaging region
    avg_region = (distances >= 0.05) & (distances <= 0.1)
    ax.axvspan(distances[avg_region][0], distances[avg_region][-1], 
               alpha=0.2, color='gray', label='Averaging region')
    
    ax.set_ylabel('Fermi Velocity (eV·Å)', fontsize=12)
    ax.set_title('Fermi Velocity near K-point', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)
    
    # Relative error
    ax = axes[1]
    relative_error = np.abs(np.array(vf_pinn_list) - np.array(vf_tb_list)) / np.array(vf_tb_list) * 100
    ax.plot(distances, relative_error, 'r-', linewidth=2)
    ax.set_xlabel('Distance from K-point (Å$^{-1}$)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('PINN vs TB Relative Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Vertical Fermi velocity plot saved: {save_path}")


def plot_training_progress_vertical(train_dir: str, save_path: str):
    """Plot training progress with 3 vertical subplots"""
    # Load training history
    history_file = os.path.join(train_dir, 'training_history.json')
    if not os.path.exists(history_file):
        print(f"✗ Training history not found: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figure with 3 vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    ax = axes[0]
    ax.semilogy(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if history['val_loss']:
        val_epochs = np.arange(5, len(history['train_loss']) + 1, 5)
        val_epochs = val_epochs[val_epochs <= len(history['val_loss']) * 5]
        ax.semilogy(val_epochs[:len(history['val_loss'])], history['val_loss'], 
                   'r-', label='Validation', linewidth=2)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Loss components
    ax = axes[1]
    components = ['data_loss', 'dirac_symmetric_loss', 'fermi_velocity_loss']
    colors = ['blue', 'orange', 'green']
    for comp, color in zip(components, colors):
        if comp in history and history[comp]:
            label = comp.replace('_loss', '').replace('_', ' ').title()
            ax.semilogy(epochs[:len(history[comp])], history[comp], 
                       color=color, label=label, linewidth=1.5)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Components', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Physics metrics
    ax = axes[2]
    if 'k_gap' in history and history['k_gap']:
        val_epochs = np.arange(5, len(history['train_loss']) + 1, 5)
        val_epochs = val_epochs[:len(history['k_gap'])]
        ax.plot(val_epochs, history['k_gap'], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gap at K-points (eV)', fontsize=12)
    ax.set_title('Physics Constraint: Gap at Dirac Points', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 2.0)
    
    plt.suptitle('SCMS-PINN v35 Training Progress', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Vertical training progress saved: {save_path}")


def plot_epoch_six_combo(model: torch.nn.Module, tb_model: TightBindingTarget,
                        train_dir: str, epoch: int, save_path: str):
    """Create six-panel combo plot for a specific epoch"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Band Structure
    ax1 = fig.add_subplot(gs[0, 0])
    k_path, labels, positions = get_band_path(n_points=300)
    k_path_tensor = torch.tensor(k_path, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pinn_output = model(k_path_tensor)
        tb_output = tb_model(k_path_tensor)
    
    ax1.plot(tb_output['valence'].cpu(), 'b-', label='TB valence', linewidth=2)
    ax1.plot(tb_output['conduction'].cpu(), 'r-', label='TB conduction', linewidth=2)
    ax1.plot(pinn_output['valence'].cpu(), 'b--', label='PINN valence', alpha=0.7)
    ax1.plot(pinn_output['conduction'].cpu(), 'r--', label='PINN conduction', alpha=0.7)
    ax1.set_title(f'Band Structure - Epoch {epoch}')
    ax1.set_xlabel('k-path')
    ax1.set_ylabel('Energy (eV)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Gap Error Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    n_grid = 50
    k_range = 2.5
    kx = np.linspace(-k_range, k_range, n_grid)
    ky = np.linspace(-k_range, k_range, n_grid)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    k_points_flat = torch.tensor(
        np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=-1),
        dtype=torch.float32, device=device
    )
    
    with torch.no_grad():
        pinn_out = model(k_points_flat)
        tb_out = tb_model(k_points_flat)
    
    gap_error = np.abs(pinn_out['gap'].cpu().numpy() - tb_out['gap'].cpu().numpy())
    gap_error_grid = gap_error.reshape(n_grid, n_grid)
    
    im = ax2.imshow(gap_error_grid, extent=[-k_range, k_range, -k_range, k_range],
                    origin='lower', cmap='hot', vmin=-2, vmax=2)
    ax2.set_title(f'Gap Error Heatmap - Epoch {epoch}')
    ax2.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax2.set_ylabel('$k_y$ (Å$^{-1}$)')
    plt.colorbar(im, ax=ax2, label='Gap Error (eV)')
    
    # Mark K-points
    K_points_np = get_k_points().cpu().numpy()
    ax2.scatter(K_points_np[:, 0], K_points_np[:, 1], c='white', s=50, marker='o')
    
    # 3. Fermi Velocity Magnitude
    ax3 = fig.add_subplot(gs[0, 2])
    K_point = K_points_np[0]
    distances = np.linspace(0.05, 0.3, 30)
    
    vf_tb = []
    vf_pinn = []
    
    for dist in distances:
        k = K_point + dist * np.array([1, 0])
        k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Calculate gradients without no_grad context
        k_tensor.requires_grad_(True)
        tb_gap = tb_model(k_tensor)['gap']
        tb_grad = torch.autograd.grad(tb_gap, k_tensor)[0]
        vf_tb.append(0.5 * torch.norm(tb_grad).item())
        
        k_tensor = k_tensor.detach().requires_grad_(True)
        pinn_gap = model(k_tensor)['gap']
        pinn_grad = torch.autograd.grad(pinn_gap, k_tensor)[0]
        vf_pinn.append(0.5 * torch.norm(pinn_grad).item())
    
    ax3.plot(distances, vf_tb, 'b-', label='TB', linewidth=2)
    ax3.plot(distances, vf_pinn, 'r--', label='PINN', linewidth=2)
    ax3.axhline(y=5.75, color='g', linestyle=':', label='Expected')
    ax3.set_title(f'Fermi Velocity Magnitude - Epoch {epoch}')
    ax3.set_xlabel('Distance from K (Å$^{-1}$)')
    ax3.set_ylabel('Velocity (eV·Å)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Directional Fermi Velocity
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    dist = 0.1
    
    vf_tb_dir = []
    vf_pinn_dir = []
    
    for angle in angles:
        k = K_point + dist * np.array([np.cos(angle), np.sin(angle)])
        k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Calculate gradients without no_grad context
        k_tensor.requires_grad_(True)
        tb_gap = tb_model(k_tensor)['gap']
        tb_grad = torch.autograd.grad(tb_gap, k_tensor)[0]
        vf_tb_dir.append(0.5 * torch.norm(tb_grad).item())
        
        k_tensor = k_tensor.detach().requires_grad_(True)
        pinn_gap = model(k_tensor)['gap']
        pinn_grad = torch.autograd.grad(pinn_gap, k_tensor)[0]
        vf_pinn_dir.append(0.5 * torch.norm(pinn_grad).item())
    
    ax4.plot(angles, vf_tb_dir, 'b-', label='TB', linewidth=2)
    ax4.plot(angles, vf_pinn_dir, 'r--', label='PINN', linewidth=2)
    ax4.set_title(f'Directional Fermi Velocity - Epoch {epoch}')
    ax4.legend(loc='upper right')
    
    # 5. Vertical Shift Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    n_samples = 5000
    k_random = torch.rand(n_samples, 2, device=device) * 4 - 2
    
    with torch.no_grad():
        pinn_out = model(k_random)
        tb_out = tb_model(k_random)
    
    v_shift = pinn_out['valence'].cpu() - tb_out['valence'].cpu()
    c_shift = pinn_out['conduction'].cpu() - tb_out['conduction'].cpu()
    
    ax5.hist(v_shift, bins=50, alpha=0.5, color='blue', label=f'Valence (mean: {v_shift.mean():.3f} eV)')
    ax5.hist(c_shift, bins=50, alpha=0.5, color='red', label=f'Conduction (mean: {c_shift.mean():.3f} eV)')
    ax5.axvline(x=v_shift.mean(), color='blue', linestyle='--')
    ax5.axvline(x=c_shift.mean(), color='red', linestyle='--')
    ax5.set_title(f'Vertical Shift Distribution - Epoch {epoch}')
    ax5.set_xlabel('Energy Shift (eV)')
    ax5.set_ylabel('Count')
    ax5.legend()
    
    # 6. Training Progress
    ax6 = fig.add_subplot(gs[1, 2])
    history_file = os.path.join(train_dir, 'training_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        epochs_range = range(1, min(epoch+1, len(history['train_loss'])+1))
        ax6.semilogy(epochs_range, history['train_loss'][:epoch], 'b-', label='Train')
        if history['val_loss']:
            val_epochs = np.arange(5, epoch+1, 5)
            val_epochs = val_epochs[val_epochs <= len(history['val_loss']) * 5]
            if len(val_epochs) > 0:
                ax6.semilogy(val_epochs[:len(history['val_loss'])], 
                           history['val_loss'][:len(val_epochs)], 'r-', label='Val')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.set_title('Training Progress')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Add epoch information
    fig.suptitle(f'SCMS-PINN v35 Analysis - Epoch {epoch}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Six-combo plot for epoch {epoch} saved: {save_path}")


def plot_c6v_symmetry_test(model: torch.nn.Module, save_path: str):
    """Test and visualize C6v symmetry at different |k| values"""
    # Test at multiple radii
    radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, radius in enumerate(radii):
        ax = axes[idx]
        angles = np.linspace(0, 2*np.pi, 360)
        gaps = []
        
        for angle in angles:
            k = radius * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                output = model(k_tensor)
                gaps.append(output['gap'].cpu().item())
        
        # Plot in polar coordinates
        ax = plt.subplot(2, 3, idx+1, projection='polar')
        ax.plot(angles, gaps, 'b-', linewidth=2)
        
        # Check C6 symmetry - should repeat every 60 degrees
        gaps_array = np.array(gaps)
        c6_segments = []
        for i in range(6):
            start = int(i * 60 * len(angles) / 360)
            end = int((i + 1) * 60 * len(angles) / 360)
            c6_segments.append(gaps_array[start:end])
        
        # Calculate C6 error
        c6_error = np.std([np.mean(seg) for seg in c6_segments])
        
        ax.set_title(f'Gap at |k| = {radius:.1f} Å$^{{-1}}$\nC6 error: {c6_error:.2e} eV')
        ax.set_ylim(0, max(gaps) * 1.1 if max(gaps) > 0 else 1)
    
    plt.suptitle('C6v Symmetry Test', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ C6v symmetry test saved: {save_path}")


def plot_energy_scale_analysis(model: torch.nn.Module, tb_model: TightBindingTarget,
                              save_path: str):
    """Analyze energy scale distribution"""
    # Sample many k-points
    n_samples = 10000
    k_random = torch.rand(n_samples, 2, device=device) * 6 - 3  # [-3, 3] range
    
    with torch.no_grad():
        pinn_out = model(k_random)
        tb_out = tb_model(k_random)
    
    # Convert to numpy
    pinn_v = pinn_out['valence'].cpu().numpy()
    pinn_c = pinn_out['conduction'].cpu().numpy()
    tb_v = tb_out['valence'].cpu().numpy()
    tb_c = tb_out['conduction'].cpu().numpy()
    
    # Energy shifts
    v_shift = pinn_v - tb_v
    c_shift = pinn_c - tb_c
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy distribution
    ax = axes[0]
    bins = np.linspace(-10, 10, 50)
    ax.hist(tb_v, bins=bins, alpha=0.3, color='blue', label='TB', density=True)
    ax.hist(tb_c, bins=bins, alpha=0.3, color='blue', density=True)
    ax.hist(pinn_v, bins=bins, alpha=0.3, color='red', label='PINN', density=True)
    ax.hist(pinn_c, bins=bins, alpha=0.3, color='red', label='PINN', density=True)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Density')
    ax.set_title('Energy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Vertical shift distribution
    ax = axes[1]
    ax.hist(v_shift, bins=50, alpha=0.5, color='blue', 
            label=f'Valence (mean: {v_shift.mean():.3f} eV)')
    ax.hist(c_shift, bins=50, alpha=0.5, color='red',
            label=f'Conduction (mean: {c_shift.mean():.3f} eV)')
    ax.axvline(x=v_shift.mean(), color='blue', linestyle='--', linewidth=2)
    ax.axvline(x=c_shift.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Energy Shift (eV)')
    ax.set_ylabel('Count')
    ax.set_title('Vertical Energy Shift Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Energy Scale Analysis v35', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Energy scale analysis saved: {save_path}")


def plot_critical_points_analysis(model: torch.nn.Module, tb_model: TightBindingTarget,
                                 save_path: str):
    """Analyze model performance at critical points like v31"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Critical points
    a = LATTICE_CONSTANT
    critical_points = {
        'Γ': np.array([0, 0]),
        'K': np.array([4*np.pi/(3*a), 0]),
        'K\'': np.array([2*np.pi/(3*a), 2*np.pi/(np.sqrt(3)*a)]),
        'M': np.array([np.pi/a, np.pi/(np.sqrt(3)*a)]),
        'M\'': np.array([0, 2*np.pi/(np.sqrt(3)*a)])
    }
    
    # 1. Energy comparison at critical points
    ax = axes[0, 0]
    point_names = list(critical_points.keys())
    tb_valence = []
    tb_conduction = []
    pinn_valence = []
    pinn_conduction = []
    
    for name, k_point in critical_points.items():
        k_tensor = torch.tensor(k_point, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            tb_out = tb_model(k_tensor)
            pinn_out = model(k_tensor)
        
        tb_valence.append(tb_out['valence'].cpu().item())
        tb_conduction.append(tb_out['conduction'].cpu().item())
        pinn_valence.append(pinn_out['valence'].cpu().item())
        pinn_conduction.append(pinn_out['conduction'].cpu().item())
    
    x = np.arange(len(point_names))
    width = 0.35
    
    ax.bar(x - width/2, tb_valence, width/2, label='TB Valence', color='blue', alpha=0.7)
    ax.bar(x, tb_conduction, width/2, label='TB Conduction', color='red', alpha=0.7)
    ax.bar(x + width/2, pinn_valence, width/2, label='PINN Valence', color='blue', alpha=0.4)
    ax.bar(x + width, pinn_conduction, width/2, label='PINN Conduction', color='red', alpha=0.4)
    
    ax.set_xlabel('Critical Points')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy at Critical Points')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(point_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Gap comparison
    ax = axes[0, 1]
    tb_gaps = np.array(tb_conduction) - np.array(tb_valence)
    pinn_gaps = np.array(pinn_conduction) - np.array(pinn_valence)
    
    ax.bar(x - width/4, tb_gaps, width/2, label='TB', color='green', alpha=0.7)
    ax.bar(x + width/4, pinn_gaps, width/2, label='PINN', color='orange', alpha=0.7)
    
    ax.set_xlabel('Critical Points')
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('Band Gap at Critical Points')
    ax.set_xticks(x)
    ax.set_xticklabels(point_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error analysis
    ax = axes[0, 2]
    valence_error = np.abs(np.array(pinn_valence) - np.array(tb_valence))
    conduction_error = np.abs(np.array(pinn_conduction) - np.array(tb_conduction))
    gap_error = np.abs(pinn_gaps - tb_gaps)
    
    ax.bar(x - width/2, valence_error, width/3, label='Valence', color='blue', alpha=0.7)
    ax.bar(x, conduction_error, width/3, label='Conduction', color='red', alpha=0.7)
    ax.bar(x + width/2, gap_error, width/3, label='Gap', color='green', alpha=0.7)
    
    ax.set_xlabel('Critical Points')
    ax.set_ylabel('Absolute Error (eV)')
    ax.set_title('Error at Critical Points')
    ax.set_xticks(x)
    ax.set_xticklabels(point_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Fermi velocity at K-points
    ax = axes[1, 0]
    K_points = [critical_points['K'], critical_points['K\'']]
    K_names = ['K', 'K\'']
    
    vf_tb = []
    vf_pinn = []
    
    for k_point in K_points:
        # Sample around K-point
        distances = np.linspace(0.01, 0.1, 20)
        vf_tb_avg = []
        vf_pinn_avg = []
        
        for dist in distances:
            angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
            vf_tb_angles = []
            vf_pinn_angles = []
            
            for angle in angles:
                k = k_point + dist * np.array([np.cos(angle), np.sin(angle)])
                k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Calculate gradients
                k_tensor.requires_grad_(True)
                tb_gap = tb_model(k_tensor)['gap']
                tb_grad = torch.autograd.grad(tb_gap, k_tensor)[0]
                vf_tb_angles.append(0.5 * torch.norm(tb_grad).item())
                
                k_tensor = k_tensor.detach().requires_grad_(True)
                pinn_gap = model(k_tensor)['gap']
                pinn_grad = torch.autograd.grad(pinn_gap, k_tensor)[0]
                vf_pinn_angles.append(0.5 * torch.norm(pinn_grad).item())
            
            vf_tb_avg.append(np.mean(vf_tb_angles))
            vf_pinn_avg.append(np.mean(vf_pinn_angles))
        
        vf_tb.append(np.mean(vf_tb_avg))
        vf_pinn.append(np.mean(vf_pinn_avg))
    
    x = np.arange(len(K_names))
    ax.bar(x - width/4, vf_tb, width/2, label='TB', color='blue', alpha=0.7)
    ax.bar(x + width/4, vf_pinn, width/2, label='PINN', color='red', alpha=0.7)
    ax.axhline(y=EXPECTED_FERMI_VELOCITY, color='green', linestyle='--', 
               label=f'Expected: {EXPECTED_FERMI_VELOCITY:.2f} eV·Å')
    
    ax.set_xlabel('K-points')
    ax.set_ylabel('Fermi Velocity (eV·Å)')
    ax.set_title('Fermi Velocity at Dirac Points')
    ax.set_xticks(x)
    ax.set_xticklabels(K_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Symmetry validation
    ax = axes[1, 1]
    # Test C3 symmetry at M point
    M_point = critical_points['M']
    angles = np.array([0, 120, 240]) * np.pi / 180
    radius = 0.1
    
    gaps_tb = []
    gaps_pinn = []
    
    for angle in angles:
        k = M_point + radius * np.array([np.cos(angle), np.sin(angle)])
        k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            tb_out = tb_model(k_tensor)
            pinn_out = model(k_tensor)
        
        gaps_tb.append(tb_out['gap'].cpu().item())
        gaps_pinn.append(pinn_out['gap'].cpu().item())
    
    x = np.arange(len(angles))
    ax.bar(x - width/4, gaps_tb, width/2, label='TB', color='blue', alpha=0.7)
    ax.bar(x + width/4, gaps_pinn, width/2, label='PINN', color='red', alpha=0.7)
    
    ax.set_xlabel('Rotation Angle')
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('C3 Symmetry Test at M Point')
    ax.set_xticks(x)
    ax.set_xticklabels(['0°', '120°', '240°'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Path integral analysis
    ax = axes[1, 2]
    # Integrate gap along high-symmetry path
    path_segments = {
        'Γ→K': (critical_points['Γ'], critical_points['K']),
        'K→M': (critical_points['K'], critical_points['M']),
        'M→Γ': (critical_points['M'], critical_points['Γ'])
    }
    
    segment_names = list(path_segments.keys())
    tb_integrals = []
    pinn_integrals = []
    
    for name, (start, end) in path_segments.items():
        n_points = 50
        t = np.linspace(0, 1, n_points)
        tb_gaps = []
        pinn_gaps = []
        
        for ti in t:
            k = (1 - ti) * start + ti * end
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                tb_out = tb_model(k_tensor)
                pinn_out = model(k_tensor)
            
            tb_gaps.append(tb_out['gap'].cpu().item())
            pinn_gaps.append(pinn_out['gap'].cpu().item())
        
        # Trapezoidal integration
        tb_integrals.append(np.trapz(tb_gaps, t))
        pinn_integrals.append(np.trapz(pinn_gaps, t))
    
    x = np.arange(len(segment_names))
    ax.bar(x - width/4, tb_integrals, width/2, label='TB', color='blue', alpha=0.7)
    ax.bar(x + width/4, pinn_integrals, width/2, label='PINN', color='red', alpha=0.7)
    
    ax.set_xlabel('Path Segment')
    ax.set_ylabel('Integrated Gap')
    ax.set_title('Path-Integrated Band Gap')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Critical Points Analysis (v35)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Critical points analysis saved: {save_path}")


def plot_angle_resolved_properties(model: torch.nn.Module, tb_model: TightBindingTarget,
                                  save_path: str):
    """Plot angle-resolved properties similar to v31"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Parameters
    K_point = get_k_points()[0].cpu().numpy()
    angles = np.linspace(0, 2*np.pi, 360)
    distances = np.array([0.05, 0.1, 0.15, 0.2])
    colors = ['blue', 'green', 'orange', 'red']
    
    # 1. Angle-resolved gap
    ax = axes[0, 0]
    for i, dist in enumerate(distances):
        gaps = []
        for angle in angles:
            k = K_point + dist * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                output = model(k_tensor)
                gaps.append(output['gap'].cpu().item())
        
        ax.plot(angles * 180/np.pi, gaps, color=colors[i], 
               label=f'r={dist:.2f} Å$^{{-1}}$', linewidth=2)
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('Angle-Resolved Band Gap near K-point')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # 2. Angle-resolved Fermi velocity
    ax = axes[0, 1]
    for i, dist in enumerate(distances):
        velocities = []
        for angle in angles:
            k = K_point + dist * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Calculate gradient without no_grad context
            k_tensor.requires_grad_(True)
            gap = model(k_tensor)['gap']
            grad = torch.autograd.grad(gap, k_tensor)[0]
            vf = 0.5 * torch.norm(grad).item()
            velocities.append(vf)
        
        ax.plot(angles * 180/np.pi, velocities, color=colors[i],
               label=f'r={dist:.2f} Å$^{{-1}}$', linewidth=2)
    
    ax.axhline(y=5.75, color='black', linestyle='--', alpha=0.5, label='Expected')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Fermi Velocity (eV·Å)')
    ax.set_title('Angle-Resolved Fermi Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # 3. Berry curvature-like quantity
    ax = axes[1, 0]
    n_grid = 100
    k_range = 1.5
    kx = np.linspace(K_point[0] - k_range, K_point[0] + k_range, n_grid)
    ky = np.linspace(K_point[1] - k_range, K_point[1] + k_range, n_grid)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    berry_like = np.zeros((n_grid, n_grid))
    
    for i in range(n_grid):
        for j in range(n_grid):
            k = np.array([kx_grid[i, j], ky_grid[i, j]])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Calculate gradients without no_grad context
            k_tensor.requires_grad_(True)
            output = model(k_tensor)
            gap = output['gap']
            
            # Compute second derivatives (related to Berry curvature)
            grad = torch.autograd.grad(gap, k_tensor, create_graph=True)[0]
            grad_x = grad[0, 0]
            grad_y = grad[0, 1]
            
            # Mixed partial derivative
            grad2_xy = torch.autograd.grad(grad_x, k_tensor, retain_graph=True)[0][0, 1]
            grad2_yx = torch.autograd.grad(grad_y, k_tensor)[0][0, 0]
            
            berry_like[i, j] = (grad2_xy - grad2_yx).item()
    
    im = ax.contourf(kx_grid, ky_grid, berry_like, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Berry-like curvature')
    ax.scatter(K_point[0], K_point[1], c='black', s=100, marker='x', linewidth=3)
    ax.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('$k_y$ (Å$^{-1}$)')
    ax.set_title('Berry Curvature-like Quantity')
    ax.set_aspect('equal')
    
    # 4. Phase diagram
    ax = axes[1, 1]
    radii = np.linspace(0, 3, 50)
    gap_min = []
    gap_max = []
    gap_avg = []
    
    for r in radii:
        if r == 0:
            k_tensor = torch.tensor([[0, 0]], dtype=torch.float32, device=device)
            with torch.no_grad():
                gap = model(k_tensor)['gap'].cpu().item()
            gap_min.append(gap)
            gap_max.append(gap)
            gap_avg.append(gap)
        else:
            gaps = []
            for angle in np.linspace(0, 2*np.pi, 36):
                k = r * np.array([np.cos(angle), np.sin(angle)])
                k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    gap = model(k_tensor)['gap'].cpu().item()
                    gaps.append(gap)
            
            gap_min.append(np.min(gaps))
            gap_max.append(np.max(gaps))
            gap_avg.append(np.mean(gaps))
    
    ax.fill_between(radii, gap_min, gap_max, alpha=0.3, color='blue', label='Min-Max range')
    ax.plot(radii, gap_avg, 'b-', linewidth=2, label='Average')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('|k| (Å$^{-1}$)')
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('Radial Gap Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Angle-Resolved Properties Analysis (v35)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Angle-resolved properties saved: {save_path}")


def main():
    """Run comprehensive visualization suite"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_visualization_v35.py <model_path> [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"visualization_v35_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(model_path)
    tb_model = TightBindingTarget()
    
    # Extract training directory
    train_dir = os.path.dirname(model_path)
    
    print("\n" + "="*60)
    print("Running Comprehensive Visualization Suite v35")
    print("="*60)
    
    # 1. Vertical band structure
    print("\n1. Creating vertical band structure plot...")
    plot_band_structure_vertical(model, tb_model, 
                                os.path.join(output_dir, 'band_structure_vertical.png'))
    
    # 2. Clean Brillouin zone
    print("2. Creating clean Brillouin zone plot...")
    plot_brillouin_zone_clean(model, tb_model,
                             os.path.join(output_dir, 'brillouin_zone_clean.png'))
    
    # 3. Vertical Fermi velocity
    print("3. Creating vertical Fermi velocity plot...")
    plot_fermi_velocity_vertical(model, tb_model,
                                os.path.join(output_dir, 'fermi_velocity_vertical.png'))
    
    # 4. Training progress
    print("4. Creating training progress plot...")
    plot_training_progress_vertical(train_dir,
                                   os.path.join(output_dir, 'training_progress_vertical.png'))
    
    # 5. Six-combo plots for epochs
    print("5. Creating six-combo plots for all checkpoint epochs...")
    # Check for checkpoint files and plot for each one found
    import glob
    checkpoint_files = sorted(glob.glob(os.path.join(train_dir, 'checkpoint_epoch_*.pth')))
    for checkpoint_path in checkpoint_files:
        # Extract epoch number from filename
        import re
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', checkpoint_path)
        if match:
            epoch = int(match.group(1))
            print(f"   Epoch {epoch}...")
            epoch_model = load_model(checkpoint_path)
            plot_epoch_six_combo(epoch_model, tb_model, train_dir, epoch,
                               os.path.join(output_dir, f'six_combo_epoch_{epoch:03d}.png'))
    
    # 6. C6v symmetry test
    print("6. Creating C6v symmetry test...")
    plot_c6v_symmetry_test(model,
                          os.path.join(output_dir, 'c6v_symmetry_test.png'))
    
    # 7. Energy scale analysis
    print("7. Creating energy scale analysis...")
    plot_energy_scale_analysis(model, tb_model,
                              os.path.join(output_dir, 'energy_scale_analysis.png'))
    
    # 8. Critical points analysis (from v31)
    print("8. Creating critical points analysis...")
    plot_critical_points_analysis(model, tb_model,
                                 os.path.join(output_dir, 'critical_points_analysis.png'))
    
    # 9. Angle-resolved properties (bonus from v31)
    print("9. Creating angle-resolved properties analysis...")
    plot_angle_resolved_properties(model, tb_model,
                                  os.path.join(output_dir, 'angle_resolved_properties.png'))
    
    print("\n" + "="*60)
    print("✓ All visualizations completed successfully!")
    print(f"Results saved in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()