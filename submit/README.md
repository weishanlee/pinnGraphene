# Physics-Informed Neural Networks for Graphene Electronic Structure

This repository contains the implementation of Scale-Consistent Multi-Stage Physics-Informed Neural Networks (SCMS-PINN) for modeling graphene's electronic band structure based on tight-binding Hamiltonians.

## Overview

The SCMS-PINN framework addresses critical challenges in applying physics-informed neural networks to quantum mechanical systems by introducing:
- Scale-consistent loss formulation preventing gradient imbalance
- Multi-stage training strategy for stable convergence
- Adaptive physics constraints for accurate band structure prediction

## Repository Structure

### `/codes/`
Implementation of the SCMS-PINN framework for graphene:
- `scms_pinn_graphene_v35.py` - Core SCMS-PINN model implementation
- `train_scms_pinn_v35.py` - Training script with multi-stage strategy
- `physics_validation_graphene_v35.py` - Physics validation and testing
- `physics_constants_v35.py` - Physical constants for graphene
- `tb_graphene.py` - Tight-binding model implementation
- `comprehensive_visualization_v35.py` - Visualization utilities
- `update_visualization_labels_v35.py` - Label updating utilities
- `comprehensive_v35_run.sh` - Complete execution script

### `/figures/`
Visualization results from the SCMS-PINN model:
- Band structure comparisons
- Fermi velocity analysis
- Error heatmaps across training epochs
- Gap analysis at high-symmetry points
- Training progress visualization
- Architecture and workflow diagrams

## Key Features

### 1. Scale-Consistent Loss Formulation
- Addresses eigenvalue magnitude disparities (meV vs eV scale)
- Prevents gradient vanishing in multi-band systems
- Ensures balanced learning across all energy bands

### 2. Multi-Stage Training Strategy
- **Stage 1**: Basic eigenvalue learning
- **Stage 2**: Orthogonality constraint incorporation
- **Stage 3**: Full physics constraints with Fermi velocity

### 3. Physics Validation
- Dirac cone verification at K and K' points
- Fermi velocity consistency (~10^6 m/s)
- Band gap analysis at M points
- Symmetry preservation validation

## Results

The SCMS-PINN model achieves:
- **99.8% accuracy** in band structure prediction
- **<2% error** in Fermi velocity calculations
- Proper Dirac cone formation at K/K' points
- Accurate band gap prediction at M points (~3 eV)

## Requirements

### System Requirements
- Python 3.8+
- CUDA 12.1 compatible GPU (recommended for training)
- 16GB+ RAM for full model training
- Linux/Unix environment

### Python Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- **PyTorch 2.3.1**: Deep learning framework with CUDA 12.1 support
- **NumPy 2.1.2**: Numerical computing
- **SciPy 1.16.0**: Scientific computing utilities
- **Matplotlib 3.10.3**: Visualization
- **SymPy 1.13.3**: Symbolic mathematics for physics calculations

To install all dependencies:
```bash
pip install -r requirements.txt
```

For GPU support, ensure you have CUDA 12.1 installed:
```bash
# Check CUDA version
nvidia-smi
nvcc --version
```

## Installation

### Option 1: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda
```bash
# Create conda environment
conda create -n pinn-graphene python=3.8
conda activate pinn-graphene

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

To reproduce the results:

```bash
cd codes/
bash comprehensive_v35_run.sh
```

This will:
1. Train the SCMS-PINN model
2. Perform physics validation
3. Generate all visualization figures

## Physical System

Graphene parameters:
- Lattice constant: a = 2.46 Å
- Nearest-neighbor hopping: t = 2.7 eV
- Brillouin zone sampling: 41×41 k-points
- High-symmetry path: Γ → M → K → Γ

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pinnGraphene2024,
  title={Scale-Consistent Multi-Stage Physics-Informed Neural Networks for Graphene Electronic Structure},
  author={Lee, Wei Shan},
  year={2024},
  journal={arXiv preprint}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions and collaborations:
- Wei Shan Lee (wslee@g.puiching.edu.mo)
- Pui Ching Middle School Macau

## Acknowledgments

This work demonstrates the application of physics-informed machine learning to quantum mechanical systems, addressing fundamental challenges in neural network training for electronic structure calculations.