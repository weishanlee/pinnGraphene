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

The SCMS-PINN solves the tight-binding Hamiltonian for graphene:
```
H = -t ∑_{<i,j>} (a†ᵢbⱼ + b†ⱼaᵢ)
```

Where:
- `t` - nearest-neighbor hopping parameter (2.7 eV)
- `a†ᵢ, aᵢ` - creation/annihilation operators on sublattice A
- `b†ⱼ, bⱼ` - creation/annihilation operators on sublattice B

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

- **Symmetry Enforcement**: Hard constraints for D6h point group operations
- **Multi-Scale Architecture**: Separate networks for low/high energy regions
- **Adaptive Loss Weighting**: Dynamic adjustment based on physics metrics
- **Self-Consistent Iteration**: Ensures physical consistency across scales
- **Comprehensive Validation**: 15+ physics-based validation metrics

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