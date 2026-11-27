# Evaluation of Inverse Physics-Informed Neural Networks for Thermal Conductivity Estimation from Transient Thermocouple Data

Academic course project. This study compares a reimplemented nonlinear least-squares baseline and an inverse PINN for estimating thermal conductivity (k), convective heat transfer coefficient (h), and heating power (P) from nine aluminum rod experiments.

## Installation

Python 3.12 was used for the original PINN experiments. From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Data and attribution

The nine CSV files contain time and excess temperatures from six thermocouples. Original experimental dataset: Tomanek and Stutts, Mendeley Data, V1, https://doi.org/10.17632/sbcf36976g.1 (CC BY 4.0, as reported in the project). The files are converted from the original spreadsheets. Original numerical implementation: https://doi.org/10.5281/zenodo.5683312. Related paper: https://doi.org/10.1016/j.applthermaleng.2020.116362.

## Reproduction

Use a separate output directory to preserve archived results.

```bash
python src/baseline_lm.py data/50al1.csv
python src/run_experiments.py --stage nine --iters 15000 --out reproduced
python src/run_experiments.py --stage stability --iters 15000 --out reproduced
python src/pinn_inverse.py data/50al1.csv --k0 100 --seed 0 --iters 50000 --out reproduced
python src/pinn_inverse.py data/50al1.csv --k0 200 --seed 0 --iters 50000 --out reproduced
```

The comparison uses nine tests, seed 0 and k0=100. Stability uses 50al1, k0=100 and 200, seeds 0–4. Extended training uses 50al1, seed 0, both initializations. There are 20 unique main runs because one run is shared. Defaults: full time domain, stride 5, two CPU threads. Extended training changes the learning-rate schedule as well as the iteration budget. The batch runner was adjusted to use only seed 0 for the nine-test stage; the numerical training code is unchanged. Direct single-run commands do not refresh the batch summary CSV.

## Repository contents

- `src/`: baseline, PINN, shared parameters and batch runner.
- `data/`: nine experimental CSV files.
- `results/`: archived summaries, histories and predictions; `summary.csv` contains 20 main runs; `baseline.csv` contains newly recomputed baseline estimates. `preliminary/` contains the legacy-domain run.
- `figures/`: existing project plots.
- `paper/`: place the final compiled report here.

## Limitations

RMSE is evaluated on the observed data, not on an independent test set. Baseline estimates are references, not ground truth. Stability and extended training concern only one test. The implemented convection initialization retains the negative exponent (-4/5) in the last Churchill–Bernstein factor for consistency with the original runs. Exact numerical reproduction may depend on library versions and hardware; archived PINN summaries record the PyTorch version. The archived PINN runs were not retrained during repository packaging.
