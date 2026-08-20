# SURF: Surrogate Using Real-time Flow

In-situ neural surrogate training for WarpX laser-plasma accelerator simulations.

## Overview

SURF trains a PyTorch neural network surrogate **during** WarpX simulation execution, directly from live particle data in memory. The framework completely bypasses disk I/O for the training data path, enabling real-time model updates as particle trajectories evolve through each plasma stage.

Once trained, the surrogate replaces the full PIC simulation, allowing for **massive parameter scans** that would otherwise require thousands of GPU-hours.

## The Compute Savings

Conventional workflows for training surrogate models of laser-plasma accelerators follow a two-phase approach:

1. **Run the full PIC simulation** → write terabytes of particle diagnostics to disk.
2. **Post-process the diagnostics** → format, normalize, and train the neural network separately.

SURF collapses the training phase into a single pass. A WarpX after-step callback extracts live particle phase-space coordinates, normalizes them with online statistics (Welford's algorithm), and performs a training step on the GPU. The simulation never waits for disk I/O.

**The Result:** A trained model that evaluates 10,000+ beam configurations in under a second, replacing thousands of hours of cluster compute time.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. TRAINING (In-Situ)                                              │
│  WarpX simulation loop (MPI across GPUs)                            │
│                                                                     │
│  ┌────────────────┐    every N steps    ┌──────────────────────────┐│
│  │  Particle data  │ ──────────────────► │  InSituTrainer           ││
│  │  in memory      │  (callback, rank 0) │  Extract → Filter → Train││
│  └────────────────┘                      └──────────────────────────┘│
│                                                                     │
│  Output: trained_models/surf_final_model.pt                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  2. INFERENCE (Parameter Scanning)                                  │
│                                                                     │
│  Load Model ──► Generate 10,000 Beam Configs ──► NN Inference      │
│                                                                     │
│  Time: ~0.5s (vs. ~800 hours on 32 GPUs for full PIC)               │
│  Speedup: > 5,000,000x                                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Physics Setup

The simulation is a **laser-wakefield accelerator** configured with the full WarpX baseline PICMI physics:

- **Boosted frame** (γ=60) for reduced simulation cost
- **Parabolic plasma density** profile with cosine density ramps per stage
- **15 acceleration stages**, each with a beam injected via `add_species_through_plane`
- **Beam gamma ramp**: γ = 1960 + 13246 × stage_index
- **PSATD solver** with multi-J algorithm (4 z-passes, 2 depositions, divE cleaning)
- **Gaussian laser pulse** (a₀=2.36) injected via laser antenna
- **Moving window** tracking the beam at velocity c

## Components

| File | Purpose |
|---|---|
| `surf_training.py` | Main in-situ framework: NN, normalizer, trainer, and WarpX setup |
| `warpx_pytorch.def` | Production Apptainer container: ROCm 7.2.3, WarpX 26.05, OpenMPI |
| `submit.sbatch` | SLURM job submission for the Virgo cluster |
| `benchmarks/offline_training.py` | Offline baseline (Control Group) measuring I/O + batch training |
| `inference/param_scan.py` | Real-time parameter scanning using the trained surrogate |

## Usage

```bash
# 1. Build the production container
apptainer build warpx_pytorch.sif warpx_pytorch.def

# 2. Train the surrogate in-situ (runs the full PIC simulation)
sbatch submit.sbatch

# 3. Run 10,000 parameter scans in < 1 second
python inference/param_scan.py --n-scans 10000
```

## Output

- `trained_models/` — Trained PyTorch model and metadata.
- `benchmarks/offline_results.json` — Wall-clock and I/O metrics for the offline approach.
- `inference/scan_results.json` — Metrics demonstrating the inference speedup.

## References

The workflow is based on the WarpX ML surrogate training pipeline described in:

- Sandberg et al., *Hybrid beamline element ML-training for surrogates in ImpactX*, IPAC'23
- Sandberg et al., *Synthesizing Particle-In-Cell Simulations through Learning and GPU Computing*, PASC '24 (Best Paper)
- WarpX documentation: [ML dataset training workflow](https://warpx.readthedocs.io/en/24.08/usage/workflows/ml_dataset_training.html)
