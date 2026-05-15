# Molecular Dynamics with Machine Learning Potentials

This diploma project explores the use of Machine Learning Potentials (MLPs) 
for modeling interatomic interactions in molecular systems with quantum-level accuracy and classical molecular dynamics efficiency.

## Overview

Traditional molecular dynamics methods face a trade-off between computational speed and accuracy:

**Classical Molecular Dynamics (MD)** is fast but relies on empirical force fields
and cannot accurately describe complex chemical interactions such as bond breaking and formation.
**Ab-initio Molecular Dynamics (AIMD)** based on quantum mechanics and Density Functional Theory (DFT) 
provides highly accurate results but requires enormous computational resources.

This project investigates how machine learning approaches can bridge this *accuracy–scalability gap*
by learning potential energy surfaces from quantum mechanical data.

## Objectives

* Study the application of machine learning models for constructing interatomic potentials from ab-initio calculations.
* Evaluate model performance on multi-particle molecular systems.
* Compare different neural network approaches for predicting molecular properties and potential energy surfaces (PES).
* Demonstrate that ML potentials can preserve quantum-level accuracy while achieving computational efficiency close to classical MD.

## Dataset

The project uses the QM9 dataset, containing approximately 134,000 small organic molecules with quantum chemical properties.

### Files

* `gdb9.sdf` — molecular structures
* `gdb9.sdf.csv` — molecular property table

## Technologies and Libraries

* **Python**
* **RDKit** — molecular structure processing and feature preparation
* **ANI Neural Networks**
* **PiNN (Pairwise Interaction Neural Network)**
* **TensorFlow / PyTorch** (depending on implementation)
* **NumPy / Pandas / Matplotlib**

## Methods

### ANI Neural Potentials

ANI-based neural networks were used to learn atomic interaction potentials and predict molecular energies
from quantum chemical data.

### PiNN / Behler–Parrinello Neural Networks

The project also applies the Behler–Parrinello Neural Network (BPNN) architecture through the PiNN framework
for constructing Potential Energy Surfaces (PES).

### Lennard–Jones Potential Reconstruction

Artificially generated two-particle systems were used to test whether neural potentials 
can recover Lennard–Jones interaction parameters.

## Results

### 1. Lennard–Jones Potential Prediction

The ANI model successfully reconstructed Lennard–Jones potential parameters with high accuracy, 
even when trained on limited datasets.

### 2. ANI vs PiNN Performance

Both methods performed well for:

* Molecular energy prediction (`u0_atom`, `h298`)
* Equilibrium molecular geometries

However, the models struggled with:

* HOMO/LUMO energy gap prediction
* Dipole moment estimation

These properties depend on global electronic distributions, which are difficult to capture using local descriptors.

### 3. Potential Energy Surface Modeling

The Behler–Parrinello architecture implemented with PiNN demonstrated reliable PES construction for organic molecules.

The trained neural potentials:

* Significantly reduced computational cost compared to DFT
* Correctly described bond breaking and bond formation energetics
* Can be applied to more advanced molecular dynamics simulations

## Conclusion

This work demonstrates that machine learning potentials provide an effective alternative 
to traditional molecular simulation methods. Neural network potentials such as ANI and PiNN can achieve near-DFT accuracy 
while maintaining the computational efficiency required for large-scale molecular dynamics simulations.

## Author

Milena Aghabekyan
Faculty of Physics — Data Processing in Physics and Artificial Intelligence

## PiNN/qm9
https://github.com/Teoroo-CMC/PiNN/ 

https://www.tensorflow.org/datasets/catalog/qm9
