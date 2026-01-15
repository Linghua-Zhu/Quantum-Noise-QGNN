# Rethinking Quantum Noise in Quantum Machine Learning

This repository contains the code and data for the paper:

**"Rethinking Quantum Noise in Quantum Machine Learning: When Noise Improves Learning"**  
Linghua Zhu, Yulong Dong, Ziyu Zhang, and Xiaosong Li

## Overview

We investigate how quantum noise affects Quantum Graph Neural Networks (QGNNs) for molecular property prediction. Through systematic experiments with 55 randomly initialized models, we discover that quantum noise produces heterogeneous, initialization-dependent responses: some models improve under noise while others degrade.

### Key Findings

- **36.4%** of models show significant performance improvement (>2%) with quantum noise
- **14.5%** experience degradation (<-2%)  
- **49.1%** remain largely unaffected (±2%)
- Strong negative correlation (r = -0.620) between baseline performance and noise benefit
- Optimal noise level at ε = 0.005 (99.5% gate fidelity)

## Repository Structure

```
.
├── qgnn_base.py              # QGNN model implementation and training
├── screening_batch.py        # Batch experiment runner
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── noise_experiment.py       # Noise simulation framework
├── data/
│   └── data_55_seeds.py      # Complete experimental data (55 models)
├── plotting/
│   ├── plot_figure2.py       # Generate Figure 2 (heterogeneity analysis)
│   └── plot_figure3.py       # Generate Figure 3 (dose-response curves)
├── results/
│   └── Screening_results.txt # Raw experimental results
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PennyLane 0.30+
- NumPy, SciPy, Matplotlib, Pandas

### Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/quantum-noise-qgnn.git
cd quantum-noise-qgnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train a Single Model

```python
from qgnn_base import train_qgnn_model, get_config

# Configure experiment
config = get_config(
    n_layers=1,
    seed=4392,
    noise_level=0.005  # ε = 0.005
)

# Train model
results = train_qgnn_model(config, output_dir='./runs/seed_4392')
```

### 2. Run Screening Experiment

```bash
# Screen multiple models at different noise levels
python screening_batch.py --n_models 55 --noise_levels 0 0.005 0.010 0.015
```

### 3. Reproduce Paper Figures

```python
# Generate Figure 2 (Heterogeneous response analysis)
python plotting/plot_figure2.py

# Generate Figure 3 (Dose-response curves)
python plotting/plot_figure3.py
```

## Dataset

We use a subset of the QM9 dataset containing 2,000 molecules. The task is to predict the HOMO-LUMO energy gap.

- **Training**: 1,600 molecules (80%)
- **Validation**: 200 molecules (10%)  
- **Test**: 200 molecules (10%)

The QM9 dataset is automatically downloaded when running the code.

## Quantum Circuit Architecture

Our QGNN uses:
- **12 qubits**: 11 for atoms + 1 master qubit for readout
- **Single-layer architecture** (L=1) for controlled depth
- **EDU (Equivariantly Diagonalizable Unitary)** quantum graph circuits
- **Bond-type-specific parameterization** (single/double/triple/aromatic)

## Noise Model

We implement a phenomenological noise model that simulates cumulative gate errors:

```
f_noisy(G, θ, ε) = (1 - p_error) · f_noiseless(G, θ) + ξ
p_error(ε) = 1 - (1 - ε)^(N_g · L)
```

Where:
- ε: per-gate error rate
- N_g: effective gate count  
- L: circuit depth
- ξ: stochastic noise term

## Experimental Data

The complete data for all 55 models is available in `data/data_55_seeds.py`:

```python
from data.data_55_seeds import data_55_seeds
import pandas as pd

df = pd.DataFrame(data_55_seeds)
# Columns: seed, baseline_r2, improvement, best_noise, best_r2
```

## Results Summary

| Category | Count | Percentage | Mean Change |
|----------|-------|------------|-------------|
| Beneficial | 20 | 36.4% | +5.8 ± 2.9% |
| Detrimental | 8 | 14.5% | -4.2 ± 2.1% |
| Marginal | 27 | 49.1% | +0.6 ± 1.1% |

**Correlation Analysis:**
- Pearson's r = -0.620 (p < 0.001)
- Models with lower baseline performance benefit more from noise
- Suggests implicit regularization mechanism

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{zhu2026rethinking,
  title={Rethinking Quantum Noise in Quantum Machine Learning: When Noise Improves Learning},
  author={Zhu, Linghua and Dong, Yulong and Zhang, Ziyu and Li, Xiaosong},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by:
- U.S. Department of Energy, Office of Science, Scientific Discovery through Advanced Computing (SciDAC) program
- Quantum Systems Accelerator (QSA)
- National Energy Research Scientific Computing Center (NERSC)

## Contact

- Linghua Zhu: [your-email@uw.edu]
- Xiaosong Li: xsli@uw.edu
- Yulong Dong: dongyl@umich.edu

## Related Publications

1. Liang, S., Zhu, L., Li, X., & Yang, C. (2025). QuGStep: Refining step size selection in gradient estimation for variational quantum algorithms. *APL Computational Physics*, 1(2), 026110.
