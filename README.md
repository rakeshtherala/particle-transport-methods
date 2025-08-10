# particle-transport-methods
Python implementation using DQMoM and QMoM for population balance equations

This repository contains Python implementations of:

1. **Direct Quadrature Method of Moments (DQMoM)** – Solves population balance equations using a direct quadrature formulation.
2. **Quadrature Method of Moments (QMoM)** – Uses inversion of moments to reconstruct distributions.

Both methods are demonstrated with a logistic growth model starting from a gamma distribution.

## Features
- Gamma distribution initialization
- Wheeler’s algorithm for moment inversion (QMoM)
- Direct evolution of quadrature nodes and weights (DQMoM)
- Analytical comparison and plotting

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt