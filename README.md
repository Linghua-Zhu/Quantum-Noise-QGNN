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


### Requirements

- Python 3.8+
- PyTorch 2.0+
- PennyLane 0.30+
- NumPy, SciPy, Matplotlib, Pandas


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

- Linghua Zhu: linghua8@uw.edu
- Xiaosong Li: xsli@uw.edu
- Yulong Dong: dongyl@umich.edu

## Related Publications

1. Liang, S., Zhu, L., Li, X., & Yang, C. (2025). QuGStep: Refining step size selection in gradient estimation for variational quantum algorithms. *APL Computational Physics*, 1(2), 026110.
