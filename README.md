# PCA_RMT

Code for the manuscript on when PCA reveals population structure, studied with random
matrix theory. This repository regenerates every figure in `results/figures/main` and
`results/figures/supplement`.

## Setup

```bash
mamba env create -f environment.yml    # or: conda env create -f environment.yml
conda activate pca_rmt
```

Run every command below from the repository root, inside this environment.

## Redraw the figures

The small tables the figures are drawn from are committed, so this works on a fresh clone:

```bash
bash results/plotting_scripts/make_all_figures.sh
```

Each figure has one script, `results/plotting_scripts/<Figure>.py`, which can also be run on
its own. Text is set in Arial when it is installed (otherwise Liberation Sans or DejaVu Sans).

## Regenerate everything from scratch

Steps 1 and 2 need a SLURM cluster. First edit `slurm_config.yaml` (account, partition,
and how to activate the conda environment in a batch job).

| Step | Command | Writes |
|---|---|---|
| 1. Simulations | `bash src/simulations/run_all_sims.sh` | `results/simulations/` |
| 2a. Sweeps | `bash src/sweeps_analysis/run_all_sweeps.sh` | `results/sweeps/` |
| 2b. Per-figure analyses | `python src/analysis/run_all_analyses.py` | `results/analysis/` |
| 3. Sweep summaries | `python src/sweeps_analysis/render_summarise_sbatch.py --local` | `results/sweep_summaries/` |
| 4. Figures | `bash results/plotting_scripts/make_all_figures.sh` | `results/figures/` |

- Wait for all of step 1 to finish before starting step 2: each sweep runs one task per
  simulation file it finds.
- Step 1 writes about 800 GB and is the slow part (100 simulations per demography, one
  array task each). Steps 2a and 2b can be submitted together.
- Step 3 runs on a laptop or login node; use `--submit` instead of `--local` to run it as
  SLURM jobs.
- Every submit script takes `--no-submit` to write the `.sbatch` files (under `logs/` and
  `slurm_runs/`) without submitting them.

## Layout

| Path | Contents |
|---|---|
| `src/simulations/` | msprime simulations; one config per demography in `configs/` |
| `src/sweeps_analysis/` | cosine similarity of PCs with the theoretical axes over a grid of sampling designs; one config per sweep in `configs/` |
| `src/analysis/` | one script per remaining figure input, plus shared modules |
| `src/threshold_and_overlap.py` | RMT detection thresholds and predicted cosine similarities |
| `results/plotting_scripts/` | one script per figure |
| `results/analysis/`, `results/sweep_summaries/` | figure inputs (tracked) |

## Experimenting

**Simulations** (`src/simulations/configs/*.yaml`): `fixed` holds the demography and sample
sizes (`Ne`, `n_per_pop` or `n_per_deme`, `K`, `L` SNPs), `sweep` the grid of the swept
parameter (Fst, split time or 1/M: range and number of points), and `cluster` the SLURM
resources (estimated automatically when left out). Set `name` to something other than
`default` to write to `results/simulations/<kind>/<name>/` instead of overwriting.

```bash
python src/simulations/run_simulation.py CONFIG --dry-run     # list what would be simulated
python src/simulations/run_simulation.py CONFIG --estimate    # memory and time per task
python src/simulations/run_simulation.py CONFIG --submit      # submit the job array
python src/simulations/run_simulation.py CONFIG               # or run it here
```

**Sweeps** (`src/sweeps_analysis/configs/*.yaml`): `simdir` is the simulation set to read,
`sampling.vary` the parameter that distinguishes the curves (`N`, `phi`, `n_per_deme`,
`deme_fraction`, ...) and its values, `sampling.fixed` what is held constant, `fixed.L` the
number of SNPs, and `n_replicates` the subsamples per grid point. Give a new sweep its own
`name`; it is written to `results/sweeps/<name>/`.

```bash
python src/sweeps_analysis/run_sweep_analysis.py CONFIG --dry-run    # list tasks and cells
python src/sweeps_analysis/run_sweep_analysis.py CONFIG --submit     # submit array + aggregate
python src/sweeps_analysis/summarise.py results/sweeps/NAME --out results/sweep_summaries/NAME
```

The last command fits the curves and writes `curves.csv`, `crossings.csv` and `points.csv`.
To have a new sweep summarised with the rest, add it to `GROUPS` in
`src/sweeps_analysis/render_summarise_sbatch.py`.
