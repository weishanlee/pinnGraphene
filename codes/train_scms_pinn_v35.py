#!/usr/bin/env python3
"""
Training script for SCMS-PINN v35 with critical fixes:
1. Disabled/fixed smoothness loss (was exploding to >100k)
2. Fixed Fermi velocity loss with finite differences (no detach)
3. Added OOM handling with checkpoint saving
4. Memory monitoring every 100 batches
5. Reduced batch size and accumulation steps
6. Adjusted loss weights with Fermi velocity ramping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use('Agg')  # v33 FIX: Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import json
import os
import sys
import gc  # v33 FIX: Garbage collection to prevent memory leaks
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback
import signal  # v33 DEBUG: For catching segfaults
import faulthandler  # v33 DEBUG: For segfault debugging

# Import model and utilities
from scms_pinn_graphene_v35 import create_scms_pinn_v35, device, get_gpu_memory_usage, get_available_gpu_memory
from tb_graphene import (
    TightBindingTarget, generate_k_path, get_k_points,
    LATTICE_CONSTANT, HOPPING_PARAMETER
)
from physics_constants_v35 import (
    EXPECTED_FERMI_VELOCITY, DELTA_K, FERMI_VELOCITY_RADIUS_MIN,
    FERMI_VELOCITY_RADIUS_MAX, SMOOTHNESS_RADIUS, SMOOTHNESS_WEIGHT
)

# TEST CONFIGURATION (10-minute test)
# Change TEST_MODE to False to use production parameters
TEST_MODE = False  # v33: Set to False for production

# Configuration with v33 fixes
if TEST_MODE:
    # Quick test configuration (runs in ~10 minutes)
    CONFIG = {
        # Training parameters
        'epochs': 30,  # v33 TEST: 30 epochs for quick test
        'batch_size': 64,  # v33 TEST: Smaller batch for speed
        'accumulation_steps': 1,  # v33 TEST: No accumulation for speed
    }
else:
    # Production configuration
    CONFIG = {
        # Training parameters
        'epochs': 300,
        'batch_size': 64,  # v33: Further reduced to prevent segfault
        'accumulation_steps': 1,  # v33: Reduced to 1 to prevent memory accumulation
    }

# Common configuration (same for test and production)
CONFIG.update({
    'learning_rate': 1e-4,  # v33 FIX: Reduced 10x to prevent gradient explosion
    'weight_decay': 1e-5,
    'grad_clip': 0.1,  # v33 FIX: Very aggressive clipping to prevent explosion
    'scheduler_T_max': 100,
    'scheduler_eta_min': 1e-7,  # v33 FIX: Lower minimum LR
    
    # Mixed precision - v33: DISABLED due to NaN issues
    'mixed_precision': False,
    
    # Model configuration
    'enforce_symmetry': True,
    
    # Sampling configuration
    'sample_radius': 3.5,
    'critical_point_ratio': 0.3,
    'boundary_samples': 100,
    
    # Loss weights (v33 FIX: Further reduced to prevent gradient explosion)
    'initial_weights': {
        'data': 5.0,  # v33 FIX: Reduced from 20 to prevent explosion
        'dirac_symmetric': 5.0,  # v33 FIX: Reduced from 20
        'anchor': 5.0,  # v33 FIX: Reduced from 20
        'dirac_center': 5.0,  # v33 FIX: Reduced from 20
        'smoothness': 0.0,  # v33: DISABLED due to instability
        'regularization': 0.01,  # v33 FIX: Reduced from 0.1
        'fermi_velocity': 0.1  # v33 FIX: Start very low
    },
    
    # Curriculum learning stages
    'curriculum_stages': [
        {'epoch': 0, 'name': 'basic', 'k_range': 1.0, 'focus_critical': 0.3},
        {'epoch': 30, 'name': 'intermediate', 'k_range': 2.0, 'focus_critical': 0.4},
        {'epoch': 60, 'name': 'advanced', 'k_range': 3.0, 'focus_critical': 0.5},
        {'epoch': 100, 'name': 'full', 'k_range': 3.5, 'focus_critical': 0.3}
    ],
    
    # Validation and checkpointing
    'val_interval': 3 if TEST_MODE else 5,  # v33 TEST: More frequent validation
    'checkpoint_interval': 5 if TEST_MODE else 10,  # v33 TEST: More frequent checkpoints
    'plot_interval': 10 if TEST_MODE else 20,  # v33 TEST: More frequent plots,
    
    # Early stopping - v33 DEBUG: Disabled to let training run full course
    'patience': 500,  # Effectively disabled for 300 epochs
    'min_delta': 1e-7,  # Smaller threshold for improvement
    
    # v33: Debug flags (disabled for production)
    'debug_mode': False,  # Set to True for detailed debugging
    'debug_interval': 10,  # Print debug info every N batches when debug_mode=True
    'track_gradients': False,  # Set to True to monitor gradient statistics
    'detect_anomalies': False,  # Set to True to detect NaN/Inf immediately when they occur
    
    # v35: Plot control options
    'skip_fermi_velocity_plot': False  # Set to True if fermi velocity plot causes hanging
})


class AdaptiveSampler:
    """K-point sampler with critical point guarantees"""
    def __init__(self, k_scale: float = 3.5):
        self.k_scale = k_scale
        self.tb_model = TightBindingTarget()
        
        # Get critical points using get_reciprocal_basis
        from tb_graphene import get_reciprocal_basis
        b1, b2 = get_reciprocal_basis()
        self.gamma = np.array([0., 0.])
        self.M = 0.5 * (b1 + b2)
        self.K1 = (2*b1 + b2) / 3
        self.K2 = (b1 + 2*b2) / 3
        
        # All K-points
        self.K_points = get_k_points().cpu().numpy()
        
        # Combined critical points
        self.critical_points = np.vstack([
            self.gamma, self.M, self.K1, self.K2,
            -self.M, -self.K1, -self.K2
        ])
        
    def sample_batch_with_critical_points(self, batch_size: int, stage: str) -> torch.Tensor:
        """Sample batch with guaranteed critical points"""
        n_critical = min(8, int(batch_size * CONFIG['critical_point_ratio']))
        n_random = batch_size - n_critical
        
        # Sample critical points
        critical_idx = np.random.choice(len(self.critical_points), n_critical, replace=True)
        critical_samples = self.critical_points[critical_idx]
        
        # Add small perturbation to avoid exact duplicates
        critical_samples += np.random.normal(0, 0.01, critical_samples.shape)
        
        # Sample random points based on stage
        if stage == 'basic':
            k_range = 1.0
        elif stage == 'intermediate':
            k_range = 2.0
        elif stage == 'advanced':
            k_range = 3.0
        else:
            k_range = self.k_scale
            
        random_samples = np.random.uniform(-k_range, k_range, (n_random, 2))
        
        # Combine
        all_samples = np.vstack([critical_samples, random_samples])
        
        return torch.tensor(all_samples, dtype=torch.float32, device=device)


class PhysicsInformedLoss:
    """v33: Fixed physics-informed loss with proper Fermi velocity"""
    def __init__(self, tb_model: TightBindingTarget):
        self.tb_model = tb_model
        self.K_points = get_k_points()
        self.fermi_velocity_computed = 0  # Track number of FV computations
        
    def compute_losses(self, k_batch: torch.Tensor, predictions: Dict[str, torch.Tensor],
                      model: nn.Module, epoch: int = 0) -> Dict[str, torch.Tensor]:
        """Compute all loss components with v33 fixes"""
        losses = {}
        
        # 1. Data loss - match TB model
        with torch.no_grad():
            tb_output = self.tb_model(k_batch)
        
        losses['data'] = F.mse_loss(predictions['energies'], tb_output['energies'])
        
        # 2. Dirac point loss - enforce zero gap at K-points
        # Sample some K-points
        n_k_samples = min(6, k_batch.shape[0] // 10)
        k_indices = torch.randperm(len(self.K_points))[:n_k_samples]
        k_sample = self.K_points[k_indices]
        
        k_predictions = model(k_sample)
        losses['dirac_symmetric'] = k_predictions['gap'].pow(2).mean()
        
        # 3. Anchor loss - correct energies at Γ and M
        gamma = torch.zeros(1, 2, device=device)
        gamma_pred = model(gamma)
        gamma_val_target = -3 * HOPPING_PARAMETER
        gamma_cond_target = 3 * HOPPING_PARAMETER
        
        losses['anchor'] = (
            (gamma_pred['valence'] - gamma_val_target).pow(2) +
            (gamma_pred['conduction'] - gamma_cond_target).pow(2)
        ).mean()
        
        # 4. Dirac center loss - all K-points should have same energy
        if n_k_samples > 1:
            k_energies = k_predictions['energies']
            k_mean = k_energies.mean(dim=0, keepdim=True)
            losses['dirac_center'] = ((k_energies - k_mean)**2).mean()
        else:
            losses['dirac_center'] = torch.tensor(0.0, device=device)
        
        # 5. Smoothness loss - v33: DISABLED by setting weight to 0
        # Keep computation for compatibility but weight is 0
        losses['smoothness'] = torch.tensor(0.0, device=device)
        
        # 6. Regularization loss
        losses['regularization'] = model.get_regularization_loss()
        
        # 7. Fermi velocity loss - v33: FIXED with finite differences
        # v35 FIX: Use autograd-based Fermi velocity loss
        losses['fermi_velocity'] = self._compute_fermi_velocity_loss_v35_autograd(
            k_batch, predictions, model
        )
        
        return losses
    
    def _compute_fermi_velocity_loss_v35_autograd(self, k_batch: torch.Tensor, 
                                                 predictions: Dict[str, torch.Tensor],
                                                 model: nn.Module) -> torch.Tensor:
        """v35: Compute Fermi velocity loss using autograd for exact gradients"""
        # Find points near K-points (but not too close)
        with torch.no_grad():
            min_distances = torch.full((k_batch.size(0),), float('inf'), device=device)
            for K in self.K_points:
                distances = torch.norm(k_batch - K, dim=1)
                min_distances = torch.minimum(min_distances, distances)
            
            mask = (min_distances > FERMI_VELOCITY_RADIUS_MIN) & \
                   (min_distances < FERMI_VELOCITY_RADIUS_MAX)
            
            # Limit number of points to avoid memory issues
            if mask.sum() > 128:
                indices = mask.nonzero(as_tuple=True)[0]
                indices = indices[torch.randperm(len(indices), device=device)[:128]]
                mask = torch.zeros_like(mask)
                mask[indices] = True
        
        if mask.sum() < 10:
            return torch.tensor(0.0, device=device)
        
        # v35 FIX: Use autograd for exact gradient computation
        # Get selected points and enable gradient tracking
        k_selected = k_batch[mask].detach().requires_grad_(True)
        
        # Forward pass to build computation graph
        gap_selected = model(k_selected)['gap']
        
        # Calculate gradient using autograd
        # Use create_graph=True to allow backpropagation through the gradient
        grad_gap = torch.autograd.grad(
            outputs=gap_selected.sum(),
            inputs=k_selected,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Fermi velocity: vF = 0.5 * |∇Gap|
        v_pred = 0.5 * torch.norm(grad_gap, dim=1)
        
        # Target Fermi velocity
        v_target = EXPECTED_FERMI_VELOCITY
        
        # Use Huber loss for robustness
        fermi_loss = F.huber_loss(v_pred, torch.full_like(v_pred, v_target), delta=1.0)
        
        self.fermi_velocity_computed = mask.sum().item()
        
        return fermi_loss


def get_current_stage(epoch: int) -> Dict:
    """Get current curriculum stage"""
    for stage in reversed(CONFIG['curriculum_stages']):
        if epoch >= stage['epoch']:
            return stage
    return CONFIG['curriculum_stages'][0]


def plot_training_progress(history: Dict, k_path_data: Tuple, predictions: Dict,
                          tb_output: Dict, epoch: int, save_path: str):
    """Plot training progress with all curves in a single plot"""
    fig = plt.figure(figsize=(12, 8))
    
    # Single plot with all loss curves
    ax = plt.subplot(1, 1, 1)
    
    # Define colors and line styles for each curve
    styles = {
        'train_loss': {'color': 'blue', 'linestyle': '-', 'linewidth': 2, 'label': 'Train Loss'},
        'val_loss': {'color': 'red', 'linestyle': '--', 'linewidth': 2, 'label': 'Validation Loss'},
        'data_loss': {'color': 'green', 'linestyle': '-.', 'linewidth': 1.5, 'label': 'Data Loss'},
        'dirac_symmetric_loss': {'color': 'orange', 'linestyle': ':', 'linewidth': 1.5, 'label': 'Dirac Symmetric Loss'},
        'anchor_loss': {'color': 'purple', 'linestyle': '-', 'linewidth': 1, 'label': 'Anchor Loss'},
        'fermi_velocity_loss': {'color': 'brown', 'linestyle': '--', 'linewidth': 1, 'label': 'Fermi Velocity Loss'}
    }
    
    # Plot train loss (available for all epochs)
    if 'train_loss' in history and history['train_loss']:
        epochs_train = np.arange(1, len(history['train_loss']) + 1)
        ax.semilogy(epochs_train, history['train_loss'], 
                   color=styles['train_loss']['color'],
                   linestyle=styles['train_loss']['linestyle'],
                   linewidth=styles['train_loss']['linewidth'],
                   label=styles['train_loss']['label'])
    
    # Plot validation loss (sparse data - only every val_interval epochs)
    if 'val_loss' in history and history['val_loss']:
        # Validation is done every val_interval epochs (typically 5)
        val_interval = 5  # This matches CONFIG['val_interval']
        epochs_val = np.arange(val_interval, len(history['val_loss']) * val_interval + 1, val_interval)
        # Only plot up to the current epoch
        valid_indices = epochs_val <= epoch
        if np.any(valid_indices):
            ax.semilogy(epochs_val[valid_indices], np.array(history['val_loss'])[valid_indices],
                       color=styles['val_loss']['color'],
                       linestyle=styles['val_loss']['linestyle'],
                       linewidth=styles['val_loss']['linewidth'],
                       label=styles['val_loss']['label'],
                       marker='o', markersize=3)
    
    # Plot component losses (available for all epochs)
    for loss_name in ['data_loss', 'dirac_symmetric_loss', 'anchor_loss', 'fermi_velocity_loss']:
        if loss_name in history and history[loss_name]:
            epochs_comp = np.arange(1, len(history[loss_name]) + 1)
            ax.semilogy(epochs_comp, history[loss_name],
                       color=styles[loss_name]['color'],
                       linestyle=styles[loss_name]['linestyle'],
                       linewidth=styles[loss_name]['linewidth'],
                       label=styles[loss_name]['label'],
                       alpha=0.8)
    
    # Set x-axis to show full range 0-300
    ax.set_xlim(0, 300)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title(f'SCMS-PINN v35 Training Progress - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10)))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')  # v33 FIX: Close all figures to prevent memory leak
    
    # v33 FIX: Force garbage collection
    gc.collect()


def fix_colorbar_label(cbar, label_text: str):
    """Fix colorbar label rotation (from v31/v32)"""
    cbar.ax.set_ylabel(label_text, rotation=270, labelpad=20)


def plot_four_panel(model: nn.Module, tb_model: TightBindingTarget, 
                    epoch: int, save_path: str, history: Dict = None):
    """Create four-panel plot: Band Structure, Fermi Velocity, Vertical Shift, Training Progress"""
    from matplotlib import gridspec
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Band Structure (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    k_path, labels, positions = generate_k_path(300)
    k_path_tensor = torch.as_tensor(k_path, dtype=torch.float32, device=device)
    
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
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)
    
    # 2. Fermi Velocity Magnitude (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    K_points_np = get_k_points().cpu().numpy()
    K_point = K_points_np[0]
    distances = np.linspace(0.05, 0.3, 30)
    
    vf_tb = []
    vf_pinn = []
    
    for dist in distances:
        k = K_point + dist * np.array([1, 0])
        k_tensor = torch.as_tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        
        # TB velocity
        k_tensor.requires_grad_(True)
        tb_gap = tb_model(k_tensor)['gap']
        tb_grad = torch.autograd.grad(tb_gap, k_tensor, create_graph=False)[0]
        vf_tb.append(0.5 * torch.norm(tb_grad).item())
        
        # PINN velocity
        k_tensor = k_tensor.detach().requires_grad_(True)
        pinn_gap = model(k_tensor)['gap']
        pinn_grad = torch.autograd.grad(pinn_gap, k_tensor, create_graph=False)[0]
        vf_pinn.append(0.5 * torch.norm(pinn_grad).item())
    
    ax2.plot(distances, vf_tb, 'b-', label='TB', linewidth=2)
    ax2.plot(distances, vf_pinn, 'r--', label='PINN', linewidth=2)
    ax2.axhline(y=5.75, color='g', linestyle=':', label='Expected')
    ax2.set_title(f'Fermi Velocity Magnitude - Epoch {epoch}')
    ax2.set_xlabel('Distance from K (Å$^{-1}$)')
    ax2.set_ylabel('Velocity (eV·Å)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Vertical Shift Distribution (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    n_samples = 5000
    k_random = torch.rand(n_samples, 2, device=device) * 4 - 2
    
    with torch.no_grad():
        pinn_out = model(k_random)
        tb_out = tb_model(k_random)
    
    v_shift = pinn_out['valence'].cpu() - tb_out['valence'].cpu()
    c_shift = pinn_out['conduction'].cpu() - tb_out['conduction'].cpu()
    
    ax3.hist(v_shift.numpy(), bins=50, alpha=0.5, color='blue', 
             label=f'Valence (mean: {v_shift.mean():.3f} eV)')
    ax3.hist(c_shift.numpy(), bins=50, alpha=0.5, color='red', 
             label=f'Conduction (mean: {c_shift.mean():.3f} eV)')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title(f'Vertical Shift Distribution - Epoch {epoch}')
    ax3.set_xlabel('Energy Shift (eV)')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # 4. Training Progress (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    if history and 'train_loss' in history and history['train_loss']:
        epochs_range = range(1, min(epoch+1, len(history['train_loss'])+1))
        ax4.semilogy(epochs_range, history['train_loss'][:epoch], 'b-', label='Train')
        if 'val_loss' in history and history['val_loss']:
            val_epochs = [i*5 for i in range(1, len(history['val_loss'])+1) if i*5 <= epoch]
            if val_epochs:
                ax4.semilogy(val_epochs[:len(history['val_loss'])], 
                           history['val_loss'][:len(val_epochs)], 'r-', label='Val')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'SCMS-PINN v35 Analysis - Epoch {epoch}', fontsize=16)
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93, bottom=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # Safe memory cleanup
    gc.collect()
    # Note: Removed torch.cuda.empty_cache() - causes fragmentation


def plot_error_heatmap(model: nn.Module, tb_model: TightBindingTarget, 
                      epoch: int, save_dir: str):
    """2D error heatmap (from v31) - MISSING in original v33"""
    n_grid = 150
    k_range = 3.0
    kx = np.linspace(-k_range, k_range, n_grid)
    ky = np.linspace(-k_range, k_range, n_grid)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    k_points = torch.tensor(
        np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=-1),
        dtype=torch.float32, device=device
    )
    
    # Compute in batches
    batch_size = 5000
    errors = []
    
    for i in range(0, len(k_points), batch_size):
        k_batch = k_points[i:i+batch_size]
        
        with torch.no_grad():
            pinn_out = model(k_batch)
            tb_out = tb_model(k_batch)
            
            error = torch.abs(pinn_out['gap'] - tb_out['gap']).cpu().numpy()
            errors.append(error)
    
    error_grid = np.concatenate(errors).reshape(n_grid, n_grid)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    # Add safety check for contourf
    error_log = np.log10(error_grid + 1e-10)
    if np.all(np.isnan(error_log)) or np.all(np.isinf(error_log)):
        # Handle degenerate case
        error_log = np.zeros_like(error_log)
    im = ax.contourf(kx_grid, ky_grid, error_log, levels=30, cmap='hot')
    cbar = plt.colorbar(im, ax=ax)
    fix_colorbar_label(cbar, 'log$_{10}$(Gap Error)')
    
    # Mark K-points
    K_points = get_k_points().cpu().numpy()
    ax.scatter(K_points[:, 0], K_points[:, 1], c='blue', s=50,
              marker='o', edgecolors='white', linewidth=1.5)
    
    ax.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('$k_y$ (Å$^{-1}$)')
    ax.set_title(f'Gap Error Heatmap - Epoch {epoch}')
    ax.set_aspect('equal')
    
    save_path = os.path.join(save_dir, f'error_heatmap_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gap_analysis(model: nn.Module, tb_model: TightBindingTarget,
                     epoch: int, save_dir: str):
    """Gap distribution analysis (from v31)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Gap along k-path
    ax = axes[0, 0]
    k_path, labels, positions = generate_k_path(1000)
    
    with torch.no_grad():
        pinn_out = model(k_path)
        tb_out = tb_model(k_path)
    
    x = np.linspace(0, 1, len(k_path))
    ax.plot(x, tb_out['gap'].cpu().numpy(), 'g-', linewidth=2, label='TB gap')
    ax.plot(x, pinn_out['gap'].cpu().numpy(), 'g--', linewidth=2, label='PINN gap')
    
    x_ticks = np.array(positions) / len(k_path)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    for pos in x_ticks:
        ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('k-path')
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('Gap Along High-Symmetry Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Gap histogram
    ax = axes[0, 1]
    n_samples = 5000
    k_random = torch.randn(n_samples, 2, device=device) * 2.5
    
    with torch.no_grad():
        pinn_gaps = model(k_random)['gap'].cpu().numpy()
        tb_gaps = tb_model(k_random)['gap'].cpu().numpy()
    
    ax.hist(tb_gaps, bins=50, alpha=0.5, label='TB', color='blue')
    ax.hist(pinn_gaps, bins=50, alpha=0.5, label='PINN', color='red')
    ax.set_xlabel('Band Gap (eV)')
    ax.set_ylabel('Count')
    ax.set_title('Gap Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Gap error scatter
    ax = axes[1, 0]
    gap_error = pinn_gaps - tb_gaps
    ax.scatter(tb_gaps, gap_error, alpha=0.3, s=1)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('TB Gap (eV)')
    ax.set_ylabel('PINN - TB Error (eV)')
    ax.set_title('Gap Error vs True Gap')
    ax.grid(True, alpha=0.3)
    
    # 4. Critical points
    ax = axes[1, 1]
    critical_gaps = []
    critical_names = ['Γ', 'M', 'K']
    
    # Define critical points
    a = LATTICE_CONSTANT
    b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
    b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
    
    critical_k = [
        torch.zeros(1, 2, device=device),  # Gamma
        torch.tensor(np.array([0.5 * (b1 + b2)]), dtype=torch.float32, device=device),  # M
        torch.tensor(np.array([(2*b1 + b2) / 3]), dtype=torch.float32, device=device)  # K
    ]
    
    with torch.no_grad():
        for k_point in critical_k:
            gap = model(k_point)['gap'].item()
            critical_gaps.append(gap)
    
    x = np.arange(len(critical_names))
    ax.bar(x, critical_gaps, color='orange', alpha=0.7)
    ax.set_xlabel('Critical Point')
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('Gap at Critical Points')
    ax.set_xticks(x)
    ax.set_xticklabels(critical_names)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Gap Analysis - Epoch {epoch}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'gap_analysis_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')  # v33 FIX: Close all figures to prevent memory issues


def plot_fermi_velocity_heatmap(model: nn.Module, tb_model: TightBindingTarget,
                               epoch: int, save_dir: str):
    """Fermi velocity heatmap near K-point (from v32)"""
    K_point = get_k_points()[0].cpu().numpy()
    n_grid = 20  # v35 FIX: Reduced from 50 to 20 to prevent hanging (400 points instead of 2500)
    r_max = 0.3
    
    x = np.linspace(-r_max, r_max, n_grid) + K_point[0]
    y = np.linspace(-r_max, r_max, n_grid) + K_point[1]
    x_grid, y_grid = np.meshgrid(x, y)
    
    k_points = torch.tensor(
        np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1),
        dtype=torch.float32, device=device
    )
    
    vf_grid = np.zeros(len(k_points))
    
    # v35 FIX: Process in batches for efficiency and add progress tracking
    batch_size = 100  # Process 100 points at a time
    with torch.no_grad():
        for batch_start in range(0, len(k_points), batch_size):
            batch_end = min(batch_start + batch_size, len(k_points))
            batch_k = k_points[batch_start:batch_end]
            
            # Calculate distances for the batch
            batch_k_cpu = batch_k.cpu().numpy()
            distances = np.sqrt((batch_k_cpu[:, 0] - K_point[0])**2 + 
                               (batch_k_cpu[:, 1] - K_point[1])**2)
            
            # Process valid points in the batch
            valid_mask = (distances > 0.01) & (distances < 0.25)
            if valid_mask.any():
                valid_k = batch_k[valid_mask]
                gaps = model(valid_k)['gap'].cpu().numpy()
                valid_distances = distances[valid_mask]
                vf_values = gaps / (2 * valid_distances)
                
                # Update the grid
                valid_indices = np.arange(batch_start, batch_end)[valid_mask]
                vf_grid[valid_indices] = vf_values
            
            # Set invalid points to NaN
            invalid_indices = np.arange(batch_start, batch_end)[~valid_mask]
            vf_grid[invalid_indices] = np.nan
    
    vf_grid = vf_grid.reshape(n_grid, n_grid)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PINN Fermi velocity - handle all-NaN case
    if np.all(np.isnan(vf_grid)):
        # If all values are NaN, create a dummy plot
        im1 = ax1.contourf(x_grid, y_grid, np.zeros_like(vf_grid), levels=20, cmap='hot')
        ax1.text(0.5, 0.5, 'No valid data', transform=ax1.transAxes,
                ha='center', va='center', fontsize=12)
    else:
        # Replace NaN with 0 for plotting (contourf doesn't handle all-NaN well)
        vf_grid_plot = np.nan_to_num(vf_grid, nan=0.0)
        im1 = ax1.contourf(x_grid, y_grid, vf_grid_plot, levels=20, cmap='hot')
    cbar1 = plt.colorbar(im1, ax=ax1)
    fix_colorbar_label(cbar1, '$v_F$ (eV·Å)')
    ax1.scatter(K_point[0], K_point[1], c='blue', s=100, marker='o', 
               edgecolors='white', linewidth=2)
    ax1.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax1.set_ylabel('$k_y$ (Å$^{-1}$)')
    ax1.set_title('PINN Fermi Velocity')
    ax1.set_aspect('equal')
    
    # TB Fermi velocity for comparison
    vf_tb_grid = np.zeros(len(k_points))
    
    # v35 FIX: Process TB in batches too for consistency and speed
    with torch.no_grad():
        for batch_start in range(0, len(k_points), batch_size):
            batch_end = min(batch_start + batch_size, len(k_points))
            batch_k = k_points[batch_start:batch_end]
            
            # Calculate distances for the batch
            batch_k_cpu = batch_k.cpu().numpy()
            distances = np.sqrt((batch_k_cpu[:, 0] - K_point[0])**2 + 
                               (batch_k_cpu[:, 1] - K_point[1])**2)
            
            # Process valid points in the batch
            valid_mask = (distances > 0.01) & (distances < 0.25)
            if valid_mask.any():
                valid_k = batch_k[valid_mask]
                gaps = tb_model(valid_k)['gap'].cpu().numpy()
                valid_distances = distances[valid_mask]
                vf_values = gaps / (2 * valid_distances)
                
                # Update the grid
                valid_indices = np.arange(batch_start, batch_end)[valid_mask]
                vf_tb_grid[valid_indices] = vf_values
            
            # Set invalid points to NaN
            invalid_indices = np.arange(batch_start, batch_end)[~valid_mask]
            vf_tb_grid[invalid_indices] = np.nan
    
    vf_tb_grid = vf_tb_grid.reshape(n_grid, n_grid)
    
    # TB Fermi velocity - handle all-NaN case
    if np.all(np.isnan(vf_tb_grid)):
        # If all values are NaN, create a dummy plot
        im2 = ax2.contourf(x_grid, y_grid, np.zeros_like(vf_tb_grid), levels=20, cmap='hot')
        ax2.text(0.5, 0.5, 'No valid data', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12)
    else:
        # Replace NaN with 0 for plotting (contourf doesn't handle all-NaN well)
        vf_tb_grid_plot = np.nan_to_num(vf_tb_grid, nan=0.0)
        im2 = ax2.contourf(x_grid, y_grid, vf_tb_grid_plot, levels=20, cmap='hot')
    cbar2 = plt.colorbar(im2, ax=ax2)
    fix_colorbar_label(cbar2, '$v_F$ (eV·Å)')
    ax2.scatter(K_point[0], K_point[1], c='blue', s=100, marker='o', 
               edgecolors='white', linewidth=2)
    ax2.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax2.set_ylabel('$k_y$ (Å$^{-1}$)')
    ax2.set_title('TB Fermi Velocity')
    ax2.set_aspect('equal')
    
    plt.suptitle(f'Fermi Velocity Near K-point - Epoch {epoch}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'fermi_velocity_heatmap_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def validate_model(model: nn.Module, val_k_points: torch.Tensor, 
                  tb_model: TightBindingTarget, loss_fn: PhysicsInformedLoss,
                  loss_weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Validate model performance"""
    model.eval()
    
    # v35 FIX: Simple validation without autograd issues
    with torch.no_grad():
        predictions = model(val_k_points)
        tb_output = tb_model(val_k_points)
        
        # Compute basic losses
        losses = {
            'data': F.mse_loss(predictions['energies'], tb_output['energies']),
            'dirac_symmetric': torch.tensor(0.0, device=device),  # Skip for validation
            'dirac_center': torch.tensor(0.0, device=device),     # Skip for validation
            'anchor': torch.tensor(0.0, device=device),           # Skip for validation
            'smoothness': torch.tensor(0.0, device=device),       # Already disabled
            'regularization': model.get_regularization_loss() if hasattr(model, 'get_regularization_loss') else torch.tensor(0.0, device=device),
            'fermi_velocity': torch.tensor(0.0, device=device)    # Skip for validation
        }
    
    # Weighted total loss
    total_loss = sum(loss_weights.get(name, 0) * loss for name, loss in losses.items())
    
    # Physics checks
    with torch.no_grad():
        K_points = get_k_points()
        k_pred = model(K_points)
        k_gap = k_pred['gap'].abs().max().item()
        
        # Fermi velocity check
        fv_points = loss_fn.fermi_velocity_computed
        
    return total_loss.item(), {
        'total': total_loss.item(),
        **{name: loss.item() for name, loss in losses.items()},
        'k_gap': k_gap,
        'fermi_velocity_points': fv_points
    }


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int, train_loss: float, val_loss: float,
                   sampler: AdaptiveSampler, loss_weights: Dict[str, float],
                   save_dir: str):
    """Save training checkpoint with enhanced error handling"""
    try:
        # v33 DEBUG: Print checkpoint info
        print(f"  [DEBUG] Saving checkpoint for epoch {epoch}...")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_weights': loss_weights,
            'config': CONFIG
        }
        
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
        
        # v33 DEBUG: Save with error handling
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
        
        # Verify the checkpoint was saved correctly
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  ✓ Checkpoint saved: {path} ({file_size:.2f} MB)")
        else:
            print(f"  ⚠️ Warning: Checkpoint file not found after save!")
            
    except Exception as e:
        print(f"  ✗ Error saving checkpoint: {e}")
        traceback.print_exc()


# v33 DEBUG: Helper functions for enhanced error catching
def check_tensor_validity(tensor: torch.Tensor, name: str, epoch: int = -1, batch: int = -1) -> bool:
    """Check if tensor contains NaN or Inf values"""
    if tensor is None:
        return True
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"\n⚠️ TENSOR VALIDITY CHECK FAILED at epoch {epoch}, batch {batch}")
        print(f"  Tensor '{name}' contains: NaN={has_nan}, Inf={has_inf}")
        print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"  Min: {tensor.min().item():.4e}, Max: {tensor.max().item():.4e}")
        return False
    return True


def debug_loss_components(losses: Dict[str, torch.Tensor], epoch: int, batch_idx: int):
    """Debug individual loss components"""
    print(f"\n[DEBUG] Loss components at epoch {epoch}, batch {batch_idx}:")
    for name, loss in losses.items():
        if loss is not None:
            is_valid = check_tensor_validity(loss, f"loss_{name}", epoch, batch_idx)
            if is_valid:
                print(f"  {name}: {loss.item():.6f}")
            else:
                print(f"  {name}: INVALID (NaN/Inf detected)")


def segfault_handler(signum, frame):
    """Handle segmentation fault with traceback"""
    print("\n" + "="*60)
    print("⚠️ SEGMENTATION FAULT DETECTED!")
    print("="*60)
    print(f"Signal: {signum}")
    print("Python stack trace:")
    traceback.print_stack(frame)
    print("="*60)
    sys.exit(139)


def train_scms_pinn_v35(model: nn.Module, train_dir: str, config: Dict = CONFIG):
    """Main training function with v35 fixes and debugging"""
    
    # v33 DEBUG: Enable fault handler for better segfault debugging
    faulthandler.enable()
    signal.signal(signal.SIGSEGV, segfault_handler)
    print("\n✓ Segfault handler installed")
    
    print("\n" + "="*60)
    # Fixed: Only show DEBUG MODE when actually in debug mode
    if config.get('debug_mode', False):
        print("Starting SCMS-PINN v35 Training (DEBUG MODE)")
    else:
        print("Starting SCMS-PINN v35 Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Debug mode: {config.get('debug_mode', False)}")
    print(f"  - Track gradients: {config.get('track_gradients', False)}")
    print(f"  - Detect anomalies: {config.get('detect_anomalies', False)}")
    print(f"  - Patience: {config['patience']}")
    
    # v33 DEBUG: Enable anomaly detection
    if config.get('detect_anomalies', False):
        torch.autograd.set_detect_anomaly(True)
        print("  ✓ Anomaly detection enabled")
    
    # Create TB model and loss function
    tb_model = TightBindingTarget()
    loss_fn = PhysicsInformedLoss(tb_model)
    
    # Create sampler
    sampler = AdaptiveSampler(config['sample_radius'])
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config['scheduler_T_max'],
                                 eta_min=config['scheduler_eta_min'])
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None
    
    # Initialize loss weights
    loss_weights = config['initial_weights'].copy()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'data_loss': [], 'dirac_symmetric_loss': [],
        'anchor_loss': [], 'dirac_center_loss': [],
        'regularization_loss': [],  # v33 FIX: Removed smoothness_loss (disabled)
        'fermi_velocity_loss': [], 'k_gap': [],
        'learning_rate': [], 'loss_weights': []
    }
    
    # Validation set
    val_k_points = sampler.sample_batch_with_critical_points(1000, 'full')
    
    # Best model tracking
    # v35 FIX: Track best K-gap for model selection instead of weighted loss
    best_val_loss = float('inf')  # Keep for reporting
    best_k_gap = float('inf')     # Use this for model selection
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        model.set_epoch(epoch)
        
        # Get current stage
        stage = get_current_stage(epoch)
        
        # v35 FIX: Adjusted Fermi velocity ramping (cap at 2.0 instead of 5.0 with autograd)
        if epoch < 50:
            loss_weights['fermi_velocity'] = 0.1  # Start very low
        elif epoch < 150:
            loss_weights['fermi_velocity'] = 0.1 + (1.0 - 0.1) * ((epoch - 50) / 100)  # Ramp to 1.0
        else:
            loss_weights['fermi_velocity'] = 1.0 + (2.0 - 1.0) * min(1.0, (epoch - 150) / 100)  # v35: Cap at 2.0
        
        # v35 FIX: Strengthen Dirac constraints with scheduling
        if epoch < 50:
            loss_weights['dirac_symmetric'] = 5.0  # Original weight
            loss_weights['dirac_center'] = 5.0
        elif epoch < 150:
            loss_weights['dirac_symmetric'] = 12.0  # Increase priority
            loss_weights['dirac_center'] = 5.0
        else:
            loss_weights['dirac_symmetric'] = 25.0  # Strong enforcement
            loss_weights['dirac_center'] = 10.0      # Also increase center weight
        
        # Training metrics
        train_losses = []
        loss_components_epoch = {name: [] for name in loss_weights.keys()}
        
        # Number of batches
        n_batches = 50  # v33: Reduced to prevent memory issues
        
        print(f"\nEpoch {epoch + 1}/{config['epochs']} - Stage: {stage['name']}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  FV weight: {loss_weights['fermi_velocity']:.2f}")
        print(f"  Memory: {get_gpu_memory_usage():.2f}/{get_gpu_memory_usage() + get_available_gpu_memory():.2f} GB")
        
        # v33 DEBUG: Extra info around problematic epochs
        if 35 <= epoch <= 45:
            print(f"  [DEBUG] Stage k_range: {stage['k_range']}, focus_critical: {stage['focus_critical']}")
        
        # Zero gradients
        optimizer.zero_grad()  # v33: Use optimizer.zero_grad() for better memory management
        
        for batch_idx in range(n_batches):
            # v33: Wrap in try-except for OOM handling
            try:
                # Sample batch
                k_batch = sampler.sample_batch_with_critical_points(
                    config['batch_size'], stage['name']
                )
                
                # v33 DEBUG: Check input validity
                if not check_tensor_validity(k_batch, "k_batch_input", epoch, batch_idx):
                    print("⚠️ Invalid input detected, skipping batch")
                    continue
                
                # Forward pass
                if config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        predictions = model(k_batch)
                        
                        # v33 DEBUG: Check predictions validity
                        if 'energies' in predictions:
                            check_tensor_validity(predictions['energies'], "predictions_energies", epoch, batch_idx)
                        if 'gap' in predictions:
                            check_tensor_validity(predictions['gap'], "predictions_gap", epoch, batch_idx)
                        
                        losses = loss_fn.compute_losses(k_batch, predictions, model, epoch)
                        
                        # v33 DEBUG: Check each loss component
                        if batch_idx == 0 and epoch % 10 == 0:  # Debug every 10 epochs
                            debug_loss_components(losses, epoch, batch_idx)
                        
                        # Weighted total loss
                        total_loss = sum(loss_weights.get(name, 0) * loss 
                                       for name, loss in losses.items())
                        total_loss = total_loss / config['accumulation_steps']
                else:
                    predictions = model(k_batch)
                    
                    # v33 DEBUG: Check predictions validity
                    if 'energies' in predictions:
                        check_tensor_validity(predictions['energies'], "predictions_energies", epoch, batch_idx)
                    if 'gap' in predictions:
                        check_tensor_validity(predictions['gap'], "predictions_gap", epoch, batch_idx)
                    
                    losses = loss_fn.compute_losses(k_batch, predictions, model, epoch)
                    
                    # v33 DEBUG: Check each loss component
                    if batch_idx == 0 and epoch % 10 == 0:  # Debug every 10 epochs
                        debug_loss_components(losses, epoch, batch_idx)
                    
                    total_loss = sum(loss_weights.get(name, 0) * loss 
                                   for name, loss in losses.items())
                    total_loss = total_loss / config['accumulation_steps']
                
                # v33 DEBUG: Enhanced NaN and loss checking
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e10:
                    print(f"\n✗ Invalid loss detected at epoch {epoch + 1}, batch {batch_idx}")
                    print(f"Total loss: {total_loss.item() if torch.isfinite(total_loss) else 'NaN/Inf'}")
                    print("Loss components:")
                    for name, loss in losses.items():
                        val = loss.item() if torch.isfinite(loss) else ('NaN' if torch.isnan(loss) else 'Inf')
                        print(f"  {name}: {val}")
                        # Check tensor statistics
                        if name == 'data' and torch.isfinite(predictions['energies'].max()):
                            print(f"    Predictions range: [{predictions['energies'].min().item():.4f}, {predictions['energies'].max().item():.4f}]")
                    # Save checkpoint and exit gracefully
                    print("Saving emergency checkpoint...")
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                  np.mean(train_losses) if train_losses else float('inf'),
                                  best_val_loss, sampler, loss_weights, train_dir)
                    print("Emergency checkpoint saved. Exiting due to invalid loss values.")
                    # Use exit code 2 to distinguish from segfault
                    sys.exit(2)
                
                # v33 DEBUG: Monitor loss explosion
                if config.get('debug_mode', False) and total_loss.item() > 1e6:
                    print(f"\n⚠ Large loss detected at epoch {epoch}, batch {batch_idx}: {total_loss.item():.2e}")
                    for name, loss in losses.items():
                        if loss.item() > 1e4:
                            print(f"  {name}: {loss.item():.2e}")
                
                # Backward pass
                if config['mixed_precision']:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % config['accumulation_steps'] == 0:
                    # Note: Removed cuda.empty_cache() - causes fragmentation
                    # Safe Python cleanup only
                    pass
                    # v33 DEBUG: Monitor gradients before clipping
                    if config.get('track_gradients', False) and batch_idx % config.get('debug_interval', 10) == 0:
                        grad_norms = []
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.data.norm(2).item()
                                grad_norms.append(grad_norm)
                                if grad_norm > 10:
                                    print(f"  ⚠ Large gradient in {name}: {grad_norm:.2e}")
                        
                        if grad_norms:
                            max_grad = max(grad_norms)
                            avg_grad = sum(grad_norms) / len(grad_norms)
                            if batch_idx % 50 == 0:  # Less frequent printing
                                print(f"  [Grad] Max: {max_grad:.2e}, Avg: {avg_grad:.2e}")
                    
                    if config['mixed_precision']:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                        optimizer.step()
                    
                    # v33 DEBUG: Check if gradients were clipped
                    if config.get('debug_mode', False) and grad_norm > config['grad_clip']:
                        if batch_idx % 50 == 0:
                            print(f"  [Clip] Gradients clipped: {grad_norm:.2e} -> {config['grad_clip']}")
                    
                    optimizer.zero_grad()  # v33: Use optimizer.zero_grad() for better memory management
                
                # v33: Memory monitoring
                if batch_idx % 100 == 0 and batch_idx > 0:
                    mem_alloc = get_gpu_memory_usage()
                    mem_total = mem_alloc + get_available_gpu_memory()
                    print(f"  [Mem] Batch {batch_idx}/{n_batches}: {mem_alloc:.2f}/{mem_total:.2f} GB")
                
                # Record losses
                train_losses.append(total_loss.item() * config['accumulation_steps'])
                for name in losses:
                    if name in loss_components_epoch:
                        loss_components_epoch[name].append(losses[name].item())
                
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'CUDA out of memory' in str(e):
                    print(f"\n\n{'='*60}")
                    print(f"✗ CUDA Out of Memory at epoch {epoch}, batch {batch_idx}")
                    print(f"{'='*60}")
                    print(str(e))
                    
                    # Save checkpoint
                    current_train_loss = np.mean(train_losses) if train_losses else float('inf')
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                  current_train_loss, best_val_loss,
                                  sampler, loss_weights, train_dir)
                    
                    # Clean up
                    if 'k_batch' in locals():
                        del k_batch
                    if 'predictions' in locals():
                        del predictions
                    # Note: Removed cuda.empty_cache() - causes fragmentation
                    gc.collect()  # Safe Python cleanup only
                    
                    print("✓ Emergency checkpoint saved. Exiting due to OOM.")
                    sys.exit(1)
                else:
                    print(f"\n✗ Runtime Error at epoch {epoch}, batch {batch_idx}")
                    print(traceback.format_exc())
                    raise e
        
        # Epoch statistics
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        for name in loss_components_epoch:
            if name == 'smoothness':  # v33 FIX: Skip smoothness (disabled)
                continue
            if loss_components_epoch[name]:
                avg_loss = np.mean(loss_components_epoch[name])
                history[f'{name}_loss'].append(avg_loss)
        
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['loss_weights'].append(loss_weights.copy())
        
        print(f"  Train loss: {avg_train_loss:.6f}")
        
        # Debug: Check for NaN or Inf
        if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
            print(f"\n⚠️ WARNING: Invalid loss detected at epoch {epoch + 1}")
            print(f"  Train loss: {avg_train_loss}")
            print(f"  Loss components: {loss_components_epoch}")
            print("  Saving emergency checkpoint...")
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                          avg_train_loss, best_val_loss,
                          sampler, loss_weights, train_dir)
            raise ValueError(f"Invalid loss value: {avg_train_loss}")
        
        # Validation
        if (epoch + 1) % config['val_interval'] == 0:
            val_loss, val_metrics = validate_model(model, val_k_points, tb_model, 
                                                  loss_fn, loss_weights)
            history['val_loss'].append(val_loss)
            history['k_gap'].append(val_metrics['k_gap'])
            
            print(f"  Val loss: {val_loss:.6f}")
            print(f"  Gap at K-points: {val_metrics['k_gap']:.6f} eV")
            print(f"  FV points used: {val_metrics['fermi_velocity_points']}")
            
            # v35 FIX: Model selection based on K-gap instead of weighted loss
            current_k_gap = val_metrics['k_gap']
            k_gap_improvement = best_k_gap - current_k_gap
            
            if k_gap_improvement > config['min_delta']:
                print(f"  ✓ K-gap improved from {best_k_gap:.6f} to {current_k_gap:.6f} eV")
                best_k_gap = current_k_gap
                best_val_loss = val_loss  # Update for reporting
                patience_counter = 0
                
                # Save best model with metadata
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'k_gap': current_k_gap,
                    'val_loss': val_loss
                }, os.path.join(train_dir, 'best_model.pth'))
                print("  ✓ New best model saved (based on K-gap)")
            else:
                patience_counter += 1
                print(f"  ⚠ No K-gap improvement (delta: {k_gap_improvement:.6f} < {config['min_delta']:.2e})")
                print(f"  Patience: {patience_counter}/{config['patience']}")
                
                # v33 DEBUG: Log detailed metrics when patience is high
                if config.get('debug_mode', False) and patience_counter > config['patience'] // 2:
                    print(f"  DEBUG - Val loss components:")
                    for key in ['data', 'dirac_symmetric', 'fermi_velocity', 'anchor']:
                        if key in val_metrics:
                            print(f"    {key}: {val_metrics[key]:.6f}")
                
                if patience_counter >= config['patience']:
                    print(f"\n{'='*60}")
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"Best K-gap: {best_k_gap:.6f} eV")
                    print(f"Current K-gap: {current_k_gap:.6f} eV")
                    print(f"No K-gap improvement for {patience_counter} epochs")
                    print(f"{'='*60}")
                    break
        
        # Checkpointing
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                          avg_train_loss, best_val_loss,
                          sampler, loss_weights, train_dir)
        
        # Plotting
        if (epoch + 1) % config['plot_interval'] == 0:
            # Generate band structure
            k_path, labels, positions = generate_k_path(1000)
            
            model.eval()
            with torch.no_grad():
                pinn_output = model(k_path)
                tb_output = tb_model(k_path)
            
            # Convert to numpy
            pinn_np = {k: v.cpu().numpy() for k, v in pinn_output.items()}
            tb_np = {k: v.cpu().numpy() for k, v in tb_output.items()}
            
            # Plot - Original v33 plot
            plot_path = os.path.join(train_dir, f'training_progress_epoch_{epoch + 1:03d}.png')
            plot_training_progress(history, (k_path, labels, positions),
                                 pinn_np, tb_np, epoch + 1, plot_path)
            print(f"  ✓ Progress plot saved: {plot_path}")
            
            # NEW: Additional plots from v31/v32 - with error handling
            # Error heatmap (from v31)
            try:
                plot_error_heatmap(model, tb_model, epoch + 1, train_dir)
                print(f"  ✓ Error heatmap saved")
            except Exception as e:
                print(f"  ⚠ Error heatmap failed: {str(e)[:100]}")
            
            # Gap analysis (from v31)
            try:
                plot_gap_analysis(model, tb_model, epoch + 1, train_dir)
                print(f"  ✓ Gap analysis saved")
            except Exception as e:
                print(f"  ⚠ Gap analysis failed: {str(e)[:100]}")
            
            # Fermi velocity heatmap (from v32)
            # v35 FIX: Add option to skip this plot if it causes issues
            if config.get('skip_fermi_velocity_plot', False):
                print(f"  → Skipping Fermi velocity heatmap (disabled in config)")
            else:
                try:
                    import time
                    start_time = time.time()
                    plot_fermi_velocity_heatmap(model, tb_model, epoch + 1, train_dir)
                    elapsed = time.time() - start_time
                    print(f"  ✓ Fermi velocity heatmap saved ({elapsed:.1f}s)")
                    if elapsed > 30:
                        print(f"  ⚠ WARNING: Fermi velocity plot took {elapsed:.1f}s - consider disabling")
                except Exception as e:
                    print(f"  ⚠ Fermi velocity heatmap failed: {str(e)[:100]}")
            
            # v35 MODIFIED: Four-panel plot every 20 epochs (replacing six-combo)
            if (epoch + 1) % 20 == 0:
                four_panel_path = os.path.join(train_dir, f'four_panel_epoch_{epoch + 1:03d}.png')
                print(f"  → Generating four-panel plot for epoch {epoch + 1}...")
                plot_four_panel(model, tb_model, epoch + 1, four_panel_path, history)
                print(f"  ✓ Four-panel plot saved: {four_panel_path}")
            
            # v33 FIX: Safe memory cleanup - gc only
            try:
                plt.close('all')
                gc.collect()  # Python cleanup only - safe
                # Note: Removed cuda.empty_cache() - causes fragmentation
            except:
                pass
            
            # v33 FIX: Save training history periodically
            try:
                history_path = os.path.join(train_dir, 'training_history.json')
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                print(f"  ✓ Training history saved")
            except Exception as e:
                print(f"  ⚠ Failed to save history: {str(e)[:100]}")
            
            # v33 FIX: Always clean up matplotlib to prevent issues
            try:
                plt.close('all')
                gc.collect()  # v33 FIX: Use global gc import
            except:
                pass
            
            model.train()
        
        # Step scheduler
        scheduler.step()
        
        # Debug: Print epoch completion
        print(f"  ✓ Epoch {epoch + 1}/{config['epochs']} completed")
        sys.stdout.flush()  # Force flush output
        
        # v33 FIX: Periodic Python garbage collection every 10 epochs
        if (epoch + 1) % 10 == 0:
            gc.collect()  # Python cleanup only - safe
            # Note: Removed cuda.empty_cache() - causes fragmentation
            print(f"  → Python memory cleanup performed")
        
        # Check if we should exit for any reason
        if epoch + 1 >= config['epochs']:
            print(f"\n✓ Reached maximum epochs: {config['epochs']}")
    
    # Save final results
    print("\n" + "="*60)
    print("Training Completed")
    print(f"Total epochs trained: {epoch + 1}")
    print("="*60)
    
    # Save training history with proper JSON serialization
    history_path = os.path.join(train_dir, 'training_history.json')
    
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
    
    history_serializable = convert_to_serializable(history)
    
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, epoch + 1,
                  avg_train_loss, best_val_loss,
                  sampler, loss_weights, train_dir)
    
    return history


def main():
    """Main training script"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = f"training_v35_{timestamp}"
    os.makedirs(train_dir, exist_ok=True)
    
    print(f"Output directory: {train_dir}")
    
    # Save configuration
    config_path = os.path.join(train_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Create model
    model = create_scms_pinn_v35(enforce_symmetry=CONFIG['enforce_symmetry'])
    
    # Train model
    try:
        print(f"\nStarting training with config:")
        print(f"  Epochs: {CONFIG['epochs']}")
        print(f"  Batch size: {CONFIG['batch_size']}")
        print(f"  Learning rate: {CONFIG['learning_rate']}")
        print(f"  Device: {device}")
        
        history = train_scms_pinn_v35(model, train_dir, CONFIG)
        print("\n✓ Training completed successfully!")
        print(f"  Final epoch: {len(history['train_loss'])}")
        print(f"  Final train loss: {history['train_loss'][-1] if history['train_loss'] else 'N/A'}")
        
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        print(f"  Current time: {datetime.now()}")
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"\n✗ Runtime error during training: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Current time: {datetime.now()}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        
        # Check for OOM
        if 'out of memory' in str(e).lower():
            print("\n⚠️ Out of memory detected. Consider reducing batch size or model size.")
        
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Current time: {datetime.now()}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    import atexit
    
    def exit_handler():
        print("\n" + "="*60)
        print("Python script exiting")
        print(f"Exit time: {datetime.now()}")
        print("="*60)
        sys.stdout.flush()
    
    atexit.register(exit_handler)
    main()