## Multiplicative interaction channels (MICs)

Code for "Decomposing the modulation of interactions between neuronal populations" (Celotto*, Sooter*, Ährlund-Richter, Jenks, Sur†, Panzeri†; NeurIPS 2026).

MICs extend the communication subspace framework from [Semedo et al. 2019 ](https://www.sciencedirect.com/science/article/pii/S0896627319300534?via%3Dihub) (reduced-rank regression between a source population X and a target population Y) to quantify how a measured, possibly high-dimensional modulator Z reshapes the X→Y interaction. The modulation is parameterized as a low-rank (CP-decomposed) bilinear tensor, so that each channel pairs a source pattern, a target pattern, and the modulator pattern that scales their interaction. A geometric decomposition then quantifies whether each channel reshapes the baseline interaction or recruits dimensions private to the source or target.

## Repository structure

- `+MIC/` – MATLAB package implementing MICs:
  - `model_wrapper.m` – hierarchical, cross-validated fitting pipeline (Algorithm 1)
  - `heldout_test.m`, `heldout_test_cstr.m` – residual-shuffle significance tests (unconstrained and geometry-constrained fits)
  - `compute_alphas.m` – geometric decomposition of MICs relative to the baseline subspaces
  - `bilinearity_test.m`, `heldout_scores.m` – bilinearity diagnostic
  - auxiliary fitting utilities
- `simulations/` – simulation scripts, organized by figure (Figs. 2, 3, S1–S8)
- `data_analysis/` – real-data analyses, organized by figure:
  - V1–V2 benchmark data (Fig. 4a–d, S9)
  - ACA–VIP data (Fig. 4e–h, S10, S11)

Paths to data and results are marked with `@@` in the scripts and must be set before running.

## Data

- ACA–VIP dual-color two-photon imaging (28 sessions, 3 mice), released with this paper: https://osf.io/2ekj4/ (CC BY 4.0)
- V1–V2 recordings of Zandvakili & Kohn (2015), previously analyzed in Semedo et al. (2019): https://crcns.org/data-sets/vc/v1v2-1

## Requirements

Developed and tested in MATLAB R2025b, with:

- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (parallelization across cross-validation folds)

## License

- Code: MIT License (see `LICENSE`)
- ACA–VIP data: CC BY 4.0
