# polystar_bodies

This repository contains supplementary material for the paper  
**"Approximation of starshaped sets using polynomials"**  
by Chiara Meroni, Jared Miller, and Mauricio Velasco ([arXiv version](URL)).

## 📁 Brief Description of the Files

Most of the computation is carried out in **Python**:

- `quadrature_S.py`: Construction of quadrature rules for the sphere.
- `harmonic_basis.py`: Construction of spherical harmonics and zonal harmonics.
- `radial_fns.py`: Code to plot a polytope from its radial function.
- `polytopal_scenarios.py` and `non_polytopal_scenarios.py`: Code to approximate starbodies and their intersection bodies, both in the polytopal and general cases.
- `triangulations_ext_3D.py`: Used for POV-Ray visualizations.

We use **Mathematica** for high-quality plots of high-degree polynomials; see `CodeForPlots.nb`. The folder `data_to_plot` contains the JSON files used in the notebook.
