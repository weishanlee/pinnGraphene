#!/usr/bin/env python3
"""
Update visualization labels to use consistent k-path notation
Γ → Σ → M → K → Λ → Γ
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime

# Import required modules
from scms_pinn_graphene_v35 import create_scms_pinn_v35, device
from physics_validation_graphene_v35 import load_model
from tb_graphene import TightBindingTarget, generate_k_path, get_k_points
from physics_constants_v35 import (
    LATTICE_CONSTANT, HOPPING_PARAMETER, EXPECTED_FERMI_VELOCITY
)


def plot_band_structure_consistent(model: torch.nn.Module, tb_model: TightBindingTarget,
                                  save_path: str):
    """Plot band structure with consistent k-path labels: Γ → Σ → M → K → Λ → Γ"""
    # Get band path using the correct function
    k_path_tensor, labels, positions = generate_k_path(n_points=300)
    
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
    
    # Add vertical lines at high-symmetry points
    for pos in positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Band Structure along Γ-Σ-M-K-Λ-Γ', fontsize=14)
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
    
    # Set x-ticks with proper labels
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels)
    
    plt.suptitle('SCMS-PINN v35 Band Structure Validation', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Band structure with consistent labels saved: {save_path}")


def plot_dirac_points_analysis(model: torch.nn.Module, tb_model: TightBindingTarget,
                              save_path: str):
    """Analyze the 6 Dirac points (K-valleys) with clear labeling"""
    # Get all 6 K-points (Dirac points)
    K_points = get_k_points()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Brillouin zone with Dirac points
    ax = axes[0, 0]
    
    # Plot hexagonal Brillouin zone
    a = LATTICE_CONSTANT
    vertices = []
    for i in range(6):
        angle = i * np.pi / 3
        r = 4 * np.pi / (3 * a)
        vertices.append([r * np.cos(angle), r * np.sin(angle)])
    vertices = np.array(vertices)
    
    # Plot hexagon
    for i in range(6):
        j = (i + 1) % 6
        ax.plot([vertices[i, 0], vertices[j, 0]], 
               [vertices[i, 1], vertices[j, 1]], 'k-', linewidth=2)
    
    # Plot Dirac points
    K_np = K_points.cpu().numpy()
    colors = ['red', 'blue', 'red', 'blue', 'red', 'blue']  # Alternating for K/K' valleys
    labels_k = ['K₁', 'K₂', 'K₃', 'K₄', 'K₅', 'K₆']
    
    for i, (k, color, label) in enumerate(zip(K_np, colors, labels_k)):
        ax.scatter(k[0], k[1], c=color, s=200, marker='o', 
                  edgecolors='black', linewidth=2, label=label, zorder=5)
        ax.annotate(label, (k[0], k[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
    ax.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
    ax.set_title('6 Dirac Points in Brillouin Zone', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Add legend explaining valleys
    ax.text(0.02, 0.98, 'Red: K valley\nBlue: K\' valley', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Gap at each Dirac point
    ax = axes[0, 1]
    
    gaps_tb = []
    gaps_pinn = []
    
    with torch.no_grad():
        tb_out = tb_model(K_points)
        pinn_out = model(K_points)
        
        gaps_tb = tb_out['gap'].cpu().numpy()
        gaps_pinn = pinn_out['gap'].cpu().numpy()
    
    x = np.arange(len(labels_k))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gaps_tb, width, label='TB', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, gaps_pinn, width, label='PINN', color='orange', alpha=0.7)
    
    ax.set_xlabel('Dirac Point', fontsize=12)
    ax.set_ylabel('Band Gap (eV)', fontsize=12)
    ax.set_title('Gap at Each Dirac Point (Should be ~0)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_k)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add values on bars
    for bar, val in zip(bars1, gaps_tb):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, gaps_pinn):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Fermi velocity at each Dirac point
    ax = axes[0, 2]
    
    vf_tb = []
    vf_pinn = []
    
    for k_point in K_points:
        # Sample around each K-point
        radius = 0.05
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        vf_tb_avg = []
        vf_pinn_avg = []
        
        for angle in angles:
            k = k_point.cpu().numpy() + radius * np.array([np.cos(angle), np.sin(angle)])
            k_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
            
            k_tensor.requires_grad_(True)
            tb_gap = tb_model(k_tensor)['gap']
            tb_grad = torch.autograd.grad(tb_gap, k_tensor)[0]
            vf_tb_avg.append(0.5 * torch.norm(tb_grad).item())
            
            k_tensor = k_tensor.detach().requires_grad_(True)
            pinn_gap = model(k_tensor)['gap']
            pinn_grad = torch.autograd.grad(pinn_gap, k_tensor)[0]
            vf_pinn_avg.append(0.5 * torch.norm(pinn_grad).item())
        
        vf_tb.append(np.mean(vf_tb_avg))
        vf_pinn.append(np.mean(vf_pinn_avg))
    
    ax.bar(x - width/2, vf_tb, width, label='TB', color='blue', alpha=0.7)
    ax.bar(x + width/2, vf_pinn, width, label='PINN', color='red', alpha=0.7)
    ax.axhline(y=EXPECTED_FERMI_VELOCITY, color='green', linestyle='--', 
               label=f'Expected: {EXPECTED_FERMI_VELOCITY:.2f} eV·Å')
    
    ax.set_xlabel('Dirac Point', fontsize=12)
    ax.set_ylabel('Fermi Velocity (eV·Å)', fontsize=12)
    ax.set_title('Fermi Velocity at Each Dirac Point', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_k)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Energy values at Dirac points
    ax = axes[1, 0]
    
    valence = []
    conduction = []
    
    with torch.no_grad():
        for k in K_points:
            k_single = k.unsqueeze(0)
            out = model(k_single)
            valence.append(out['valence'].cpu().item())
            conduction.append(out['conduction'].cpu().item())
    
    ax.plot(x, valence, 'b-o', label='Valence', linewidth=2, markersize=8)
    ax.plot(x, conduction, 'r-s', label='Conduction', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Dirac Point', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Band Energies at Dirac Points', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_k)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Valley symmetry check
    ax = axes[1, 1]
    
    # Group by valleys (K and K')
    K_valley_gaps = [gaps_pinn[i] for i in [0, 2, 4]]  # K₁, K₃, K₅
    Kprime_valley_gaps = [gaps_pinn[i] for i in [1, 3, 5]]  # K₂, K₄, K₆
    
    valley_data = [K_valley_gaps, Kprime_valley_gaps]
    valley_labels = ['K valley\n(K₁, K₃, K₅)', 'K\' valley\n(K₂, K₄, K₆)']
    
    bp = ax.boxplot(valley_data, labels=valley_labels, patch_artist=True)
    colors_box = ['red', 'blue']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax.set_ylabel('Band Gap (eV)', fontsize=12)
    ax.set_title('Valley Symmetry Check', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 6. Critical points summary
    ax = axes[1, 2]
    
    # Get critical points from standard path
    critical_points = {
        'Γ': torch.tensor([[0, 0]], dtype=torch.float32, device=device),
        'M': None,  # Will compute
        'K': K_points[0:1]  # First K-point
    }
    
    # Compute M point
    b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
    b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
    M = 0.5 * (b1 + b2)
    critical_points['M'] = torch.tensor([M], dtype=torch.float32, device=device)
    
    point_names = ['Γ', 'M', 'K']
    gaps_critical = []
    
    for name in point_names:
        k_point = critical_points[name]
        with torch.no_grad():
            out = model(k_point)
            gap = out['gap'].cpu().item()
            gaps_critical.append(gap)
    
    expected_gaps = [
        6 * HOPPING_PARAMETER,  # Γ
        2 * HOPPING_PARAMETER,  # M
        0.0  # K
    ]
    
    x_crit = np.arange(len(point_names))
    ax.bar(x_crit - width/2, expected_gaps, width, label='Expected', color='green', alpha=0.7)
    ax.bar(x_crit + width/2, gaps_critical, width, label='PINN', color='orange', alpha=0.7)
    
    ax.set_xlabel('Critical Point', fontsize=12)
    ax.set_ylabel('Band Gap (eV)', fontsize=12)
    ax.set_title('Gap at High-Symmetry Points', fontsize=14)
    ax.set_xticks(x_crit)
    ax.set_xticklabels(point_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dirac Points (K-valleys) Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Dirac points analysis saved: {save_path}")


def main():
    """Run updated visualizations with consistent notation"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python update_visualization_labels_v35.py <model_path> [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else f"viz_consistent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(model_path)
    tb_model = TightBindingTarget()
    
    print("\n" + "="*60)
    print("Creating Visualizations with Consistent k-path Notation")
    print("="*60)
    
    # 1. Band structure with consistent labels
    print("\n1. Creating band structure with Γ-Σ-M-K-Λ-Γ labels...")
    plot_band_structure_consistent(model, tb_model, 
                                  os.path.join(output_dir, 'band_structure_consistent.png'))
    
    # 2. Dirac points analysis with clear labeling
    print("2. Creating Dirac points (K-valleys) analysis...")
    plot_dirac_points_analysis(model, tb_model,
                              os.path.join(output_dir, 'dirac_points_analysis.png'))
    
    print("\n" + "="*60)
    print("✓ Visualizations with consistent notation completed!")
    print(f"Results saved in: {output_dir}")
    print("="*60)
    
    print("\nNotation Summary:")
    print("- k-path: Γ → Σ → M → K → Λ → Γ")
    print("- Dirac points: K₁, K₂, K₃, K₄, K₅, K₆ (6 corners of hexagonal BZ)")
    print("- Valleys: K valley (K₁, K₃, K₅) and K' valley (K₂, K₄, K₆)")
    print("- Critical points: Γ (center), M (edge center), K (corner)")


if __name__ == "__main__":
    main()