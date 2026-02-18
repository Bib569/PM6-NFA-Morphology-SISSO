# Interpretable Morphology–Property Relationships in PM6:NFA Organic Solar Cells via MD Simulations and Symbolic Regression

This repository contains datasets, analysis scripts, force field parameters, and machine-learning model outputs for establishing interpretable structure–property relationships in polymer:non-fullerene acceptor (NFA) organic photovoltaic blends.

## Overview

We employ all-atom molecular dynamics (MD) simulations of 10 PM6:NFA blend morphologies to compute nine physics-based morphology descriptors, then apply the **SISSO** (Sure Independence Screening and Sparsifying Operator) symbolic regression framework to discover closed-form analytic relationships linking morphology to photovoltaic performance metrics: **J<sub>sc</sub>**, **V<sub>oc</sub>**, **FF**, and **PCE**.

## Systems Studied

| Device | NFA Acceptor | System |
|--------|-------------|--------|
| 1 | 2BO-ACl | PM6:2BO-ACl |
| 2 | 2BO-SCl | PM6:2BO-SCl |
| 3 | BO-EH-ACl | PM6:BO-EH-ACl |
| 4 | BO-EH-SCl | PM6:BO-EH-SCl |
| 5 | BTP-4F-P2EH | PM6:BTP-4F-P2EH |
| 6 | BTP-ec9 | PM6:BTP-ec9 |
| 7 | C11-BO-ACl | PM6:C11-BO-ACl |
| 8 | C11-BO-SCl | PM6:C11-BO-SCl |
| 9 | CH6 | PM6:CH6 |
| 10 | L8-BO | PM6:L8-BO |

## Morphology Descriptors

Nine descriptors were retained after multicollinearity screening (VIF < 5):

| Descriptor | Category | Description |
|-----------|----------|-------------|
| Repulsion | Interaction | Lennard-Jones repulsive energy |
| Sphericity | Geometry | Shape anisotropy of NFA packing |
| Free_Volume | Geometry | Void volume accessible to probes |
| Silhouette_Score | Packing | Cluster quality metric |
| Inertia | Geometry | Moment of inertia of NFA clusters |
| Backbone_Angle | Packing | Backbone intersection angle |
| Plane_Angle | Packing | Plane intersection angle |
| Mean_Distance | Geometry | Mean centroid–centroid distance |
| Density | Geometry | Mass density of the blend |

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/                              # Datasets
│   ├── Combined_dataset.csv           # Master dataset (10 devices × 13 columns)
│   ├── Extended_descriptors_*_dataset.csv  # Per-target descriptor files
│   ├── vif/                           # VIF analysis data
│   │   ├── Descriptor_VIF_initial_results.csv
│   │   ├── Descriptor_VIF_final_results.csv
│   │   ├── Descriptor_VIF_iteration_history.csv
│   │   └── Descriptor_VIF_removed.csv
│   └── simulation_info/               # MD simulation parameters per system
│
├── analysis/                          # Analysis scripts
│   ├── vif_analysis.py                # Variance Inflation Factor screening
│   ├── pearson_correlation.py         # Pearson correlation grid plots
│   ├── mutual_information.py          # Mutual Information + KDE analysis
│   ├── shap_analysis.py              # SHAP feature importance for SISSO models
│   ├── sisso_correlation_plots.py     # SISSO predicted vs experimental plots
│   └── fortran/                       # Geometry descriptor computation
│       ├── centroid_distance.f90      # Centroid distance calculator
│       └── angles2.f90               # Backbone/plane angle calculator
│
├── sisso_models/                      # SISSO symbolic regression results
│   ├── Jsc/
│   │   ├── feature_space/             # SIS-selected features
│   │   └── models/                    # Trained models (dim 1–5)
│   ├── Voc/
│   ├── FF/
│   └── PCE/
│
└── force_fields/                      # MD simulation parameters
    ├── ddec_charges/                  # DDEC6 partial atomic charges
    │   ├── NFAs/                      # Per-NFA charge files (.xyz)
    │   └── PM6/                       # PM6 dimer charges (.xyz)
    └── pm6_<NFA>/                     # Per-system topology files
        ├── topol.top                  # GROMACS topology
        └── *.itp                      # Molecule-level parameters
```

## Running the Analysis

### Prerequisites

```bash
pip install -r requirements.txt
```

### Analysis Scripts

All scripts should be run from the `analysis/` directory:

```bash
cd analysis

# 1. VIF multicollinearity screening
python vif_analysis.py

# 2. Pearson correlation analysis
python pearson_correlation.py

# 3. Mutual information analysis
python mutual_information.py

# 4. SHAP feature importance
python shap_analysis.py

# 5. SISSO model correlation plots
python sisso_correlation_plots.py
```

### Fortran Codes

The Fortran programs compute geometry descriptors from MD trajectories. Compile with:

```bash
cd analysis/fortran
gfortran -O2 -o centroid_distance centroid_distance.f90
gfortran -O2 -o angles2 angles2.f90
```

> **Note:** The Fortran codes contain hardcoded system parameters (e.g., `n_polymer`, `n_nfa`, `atoms_per_polymer`, `atoms_per_nfa`) that must be adapted for each PM6:NFA system. See `data/simulation_info/` for per-system values.

## MD Trajectories

Production trajectory files (`.gro`, ~20 MB each) are not included in this repository due to size constraints. They are available upon request from the authors.

## Software

- **MD Simulations:** GROMACS 2023
- **Force Field Generation:** Sobtop v1.0, ORCA 5.0
- **Charge Analysis:** DDEC6
- **Symbolic Regression:** SISSO++ 
- **Analysis:** Multiwfn, Fortran, Python 3.9+ (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SHAP)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{das2026morphology,
  author = {Das, Bibhas; Sewak, Ram and Mondal, Anirban},
  title = {Interpretable Morphology--Property Relationships in PM6:NFA Organic Solar Cells via MD Simulations and Symbolic Regression},
  journal = {},
  year = {2026},
  volume = {},
  pages = {},
  doi = {},
  publisher = {}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

For questions, issues, or collaborations:

- **Anirban Mondal** (Principal Investigator): amondal@iitgn.ac.in
- **Bibhas Das** (Lead Developer): dasbibhas@iitgn.ac.in
- **Ram Sewak** (Lead Developer): ram.sewak@iitgn.ac.in

**GitHub Issues**: [Open an issue](https://github.com/Bib569/PM6-NFA-Morphology-SISSO/issues)

## Acknowledgments

We gratefully acknowledge:

- **Indian Institute of Technology Gandhinagar** for research facilities and financial support
- **PARAM Ananta** for computational resources
- The open-source communities behind GROMACS, SISSO++, Sobtop, Multiwfn, and Python scientific libraries

---

<p align="center">
  <i>Advancing interpretable morphology–property relationships for next-generation organic photovoltaics</i>
</p>
