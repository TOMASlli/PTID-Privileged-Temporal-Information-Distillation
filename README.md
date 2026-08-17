# PTID: Privileged Temporal Information Distillation

This repository contains the core method implementation associated with
*Separating Balanced and Unbalanced Motions from Sea Surface Height via
Temporal Knowledge Distillation*.

The code is organized around two method components:

1. the temporal Teacher, single-frame Student, and auxiliary attention-transfer mechanism; and
2. the raw frequency-wavenumber cutoff used to separate balanced motions (BM) and unbalanced motions (UBM).

## Repository structure

.
|-- README.md
|-- requirements.txt
|-- ptid/
|   |-- __init__.py
|   |-- model.py
|   `-- distillation.py
`-- preprocessing/
    |-- __init__.py
    `-- raw_decomposition.py
