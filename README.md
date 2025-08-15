# Symmetry-Constrained Multi-Scale Physics-Informed Neural Networks for Graphene Electronic Band Structure

Paper announced arXiv preprint: arXiv:2508.10718 [cs.LG] 
https://arxiv.org/abs/2508.10718

## Overview

This repository contains the implementation of Symmetry-Constrained Multi-Scale Physics-Informed Neural Networks (SCMS-PINNs) for predicting the electronic band structure of graphene with unprecedented accuracy. The project demonstrates how incorporating physical symmetries and multi-scale modeling can significantly enhance the performance of PINNs in quantum mechanical applications.

## Key Features

- **Symmetry Constraints**: Enforces D6h point group symmetry of graphene lattice
- **Multi-Scale Modeling**: Adaptive basis functions for different energy scales
- **High Accuracy**: Achieves <2% error in band structure prediction
- **GPU Acceleration**: Fully optimized for CUDA-enabled GPUs
- **Comprehensive Validation**: Extensive physics-based validation metrics
- **Self-Consistent Training**: Iterative refinement with consistency checks

## Mathematical Formulation

### Tight-Binding Model
The SCMS-PINN is designed to learn the electronic band structure of graphene, which in the tight-binding approximation is given by:

```
E_TB(k) = ±t√[3 + 2cos(√3k_y a) + 4cos(√3k_y a/2)cos(3k_x a/2)]
```

Where:
- `t ≈ 2.7 eV` - nearest-neighbor hopping parameter
- `a = 2.46 Å` - lattice constant
- The ± signs correspond to conduction/valence bands

### Physics-Informed Feature Extraction
The neural network transforms raw k-points (k_x, k_y) into physics-aware features:

```
h₁ = φ_physics(k_x, k_y) = [|k|, θ_k, {|k-K_i|}₆, {cos(nθ_k)}₆, {sin(nθ_k)}₆]
```

These 31 features encode:
- Radial distance `|k|` and polar angle `θ_k`
- Distances to six K points (Dirac points)
- Six-fold rotational harmonics for C₆ᵥ symmetry

### Key Theorems

#### Theorem 1: Multi-Head ResNet Learning Capability
A multi-head neural network with physics-informed features, three specialized ResNet-6 heads (width ≥256), and adaptive blending can approximate any continuous band structure E(k) to arbitrary accuracy ε>0.

#### Theorem 2: Exact Symmetry Preservation
The group averaging operation guarantees exact C₆ᵥ symmetry:
```
E_sym(k) = (1/12) Σ_{g∈C₆ᵥ} E_NN(R_g k)
```
This ensures crystallographic symmetry independent of network architecture.

#### Theorem 3: Progressive Dirac Point Convergence
With progressive weight scheduling ω_K = {5, 12, 25} at epochs {0, 50, 150}, the expected Dirac loss converges as:
```
E[L_Dirac(T)] ≤ C/√T + ε_approx
```

### Loss Function Design
The training employs a multi-component loss with progressive strengthening:

```
L_total = L_data + ω_K(t)L_Dirac + ω_FV(t)L_Fermi + L_anchor + L_reg
```

Where:
- `L_Dirac` enforces zero gap at K points
- `L_Fermi` constrains Fermi velocity v_F ≈ 10⁶ m/s
- Progressive weights prevent training instabilities

## Repository Structure

### /codes
- `scms_pinn_graphene_v35.py` - Main SCMS-PINN implementation
- `train_scms_pinn_v35.py` - Training script with monitoring
- `physics_validation_graphene_v35.py` - Comprehensive physics validation
- `physics_constants_v35.py` - Physical constants and parameters
- `comprehensive_visualization_v35.py` - Visualization tools
- `comprehensive_v35_run.sh` - Complete pipeline execution script

### /figures
Contains generated visualizations including:
- Band structure evolution during training
- Fermi velocity heatmaps
- Error distribution analysis
- Gap analysis at high-symmetry points
- Training convergence curves
- Comparative plots with analytical solutions

### /output
- Trained models and checkpoints
- Training logs and metrics
- Validation reports
- LaTeX paper source files

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)
- PyTorch 2.3+ with CUDA support

### Installation
```bash
git clone https://github.com/weishanlee/pinnGraphene.git
cd pinnGraphene
pip install -r requirements.txt
```

### Running the Code
```bash
# Run the complete training and validation pipeline
cd codes
./comprehensive_v35_run.sh

# Or run individual components:
# Training only
python train_scms_pinn_v35.py

# Validation only (requires trained model)
python physics_validation_graphene_v35.py <model_path> <output_dir>

# Generate visualizations
python comprehensive_visualization_v35.py
```

## Results

The SCMS-PINN model achieves:
- **Band Structure Error**: < 2% across entire Brillouin zone
- **Dirac Point Accuracy**: < 0.01 eV deviation
- **Fermi Velocity**: 1.06 × 10⁶ m/s (matches experimental value)
- **Training Time**: ~1 hour on NVIDIA V100 GPU
- **Inference Speed**: Real-time predictions

## Methodology Highlights

### Multi-Head Architecture
Each specialized head employs ResNet-6 architecture with residual blocks:
```
h_{i+1} = h_i + F_i(h_i; θ_i)
```
- **K-Head**: Captures linear dispersion near Dirac points
- **M-Head**: Models saddle point behavior at M point  
- **General Head**: Smooth interpolation across Brillouin zone

### Adaptive Blending
Dynamic combination of head outputs:
```
E(k) = w_K(k)E^K(k) + w_M(k)E^M(k) + w_G(k)E^G(k)
```
Transitions from soft weighting to hard assignment at epoch 150.

### Progressive Training Schedule
- **Epochs 0-50**: ω_K=5, mild Dirac constraint for exploration
- **Epochs 50-150**: ω_K=12, head specialization phase
- **Epochs 150+**: ω_K=25, strong physics enforcement

### Exact Symmetry Enforcement
- Post-processing with all 12 C₆ᵥ operations (6 rotations + 6 reflections)
- Guarantees crystallographic symmetry without training instability
- Decouples learning from symmetry constraints

## Key Innovations

1. **Symmetry-Constrained Architecture**: First PINN to fully incorporate graphene's crystallographic symmetries
2. **Multi-Scale Decomposition**: Novel approach to handle disparate energy scales
3. **Physics-Based Loss Functions**: Custom losses derived from quantum mechanics
4. **Self-Consistent Training**: Iterative refinement with consistency checks

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lee2025scmspinn,
  title={Symmetry-Constrained Multi-Scale Physics-Informed Neural Networks for Graphene Electronic Band Structure Prediction},
  author={Lee, Wei Shan and Kwok, I Hang and Leong, Kam Ian and Chau, Chi Kiu Althina and Sio, Kei Chon},
  journal={arXiv preprint arXiv:2508.10718},
  year={2025},
  url={https://arxiv.org/abs/2508.10718}
}
```

## Authors

- Wei Shan Lee* (Corresponding Author, wslee@g.puiching.edu.mo) - Pui Ching Middle School Macau
- I Hang Kwok - Pui Ching Middle School Macau
- Kam Ian Leong - Pui Ching Middle School Macau  
- Chi Kiu Althina Chau - Pui Ching Middle School Macau
- Kei Chon Sio - University of Toronto Mississauga

## License

This project is open source and available under the MIT License.

## Acknowledgments

This work was developed using computational resources from Pui Ching Middle School Macau. We acknowledge the open-source community for the development of physics-informed neural network frameworks.

---

For questions, issues, or contributions, please open an issue on GitHub or contact the corresponding author.