# QSVM-nator

**Comparing SVM Approaches: Classical Efficiency vs. Quantum Advantage**

## Overview

QSVM-nator is a comprehensive tool for comparing classical Support Vector Machines (SVMs) with their quantum counterparts (QSVMs). This project explores how quantum computing can potentially overcome traditional SVM limitations in high-dimensional spaces and complex feature engineering.

## Key Features

- Implementation of both classical SVMs and quantum SVMs
- Comparative analysis across multiple datasets
- Benchmarking of performance metrics between classical and quantum approaches

## Requirements

- Python 3.8+
- Qiskit (IBM's quantum computing framework)
- scikit-learn
- numpy
- matplotlib
- pandas
- python-dotenv (Optional - if you want to run the program on a real quantum computer through IBM Quantum Runtime Service)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qsvm-nator.git
cd qsvm-nator

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional - setup IBM Quantum Runtime
# Assuming you have saved your runtime token as IBM_RUNTIME_TOKEN in .evn file
python3 save-runtime.py
```

## Usage

```python
# Coming soon
```

## Datasets

The project includes:
- Standard benchmark datasets
- A specially curated dataset challenging for classical kernels but separable with quantum kernels
- Generated pattern datasets following methodology from research literature

## Project Structure

```
qsvm-nator/
├── datasets/           # Dataset storage
├── plots/              # Generated plots
├── examples/           # Example scripts
├── tests/              # Unit tests
├── requirements.txt    # Package dependencies
├── main.py             # Program entry
├── utils.py            # Utilities
├── save-runtime.py     # IBM Runtime setup (optional)
├── LICENSE             # License file
└── README.md           # This file
```

## Contribution

This is a study project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IBM Qiskit team for the quantum computing framework
- Research papers that inspired this work:
  - [Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019)](https://www.nature.com/articles/s41586-019-0980-2)
  - [Schuld, M., Killoran, N. Quantum Machine Learning in Feature Hilbert Spaces. Phys. Rev. Lett. 122, 040504 (2019)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504)