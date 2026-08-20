# SURF: Surrogate Using Real-time Flow

In-situ neural surrogate training for WarpX laser-plasma accelerator simulations.

## Overview

SURF trains a PyTorch neural network surrogate **during** WarpX simulation execution, directly from live particle data in memory. The framework completely bypasses disk I/O for the training data path, enabling real-time model updates as particle trajectories evolve through each plasma stage.

## Key Idea

Conventional workflows for training surrogate models of laser-plasma accelerators follow a two-phase approach:

1. **Run the full PIC simulation** → write terabytes of particle diagnostics to disk
2. **Post-process the diagnostics** → format, normalize, and train the neural network separately

SURF collapses these into a single pass: a WarpX after-step callback extracts live particle phase-space coordinates from memory, normalizes them with online statistics (Welford's algorithm), and performs a training step on the GPU surrogate model. The simulation never waits for disk I/O.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  WarpX simulation loop (MPI across GPUs)                │
│                                                         │
│  ┌────────────────┐    every N steps    ┌──────────────┐│
│  │  Particle data  │ ──────────────────► │  InSituTrainer││
│  │  in memory      │  (callback, rank 0) │              ││
│  └────────────────┘                      │  1. Extract  ││
│                                          │  2. Filter   ││
│                                          │  3. Normalize││
│                                          │  4. Train NN  ││
│                                          └──────────────┘│
│                                                         │
│  GPU: WarpX PIC solver + PyTorch surrogate (same GPU)    │
└─────────────────────────────────────────────────────────┘
```

## Physics Setup

The simulation is a **laser-wakefield accelerator** configured with the full WarpX baseline PICMI physics:

- **Boosted frame** (γ=60) for reduced simulation cost
- **Parabolic plasma density** profile with cosine density ramps per stage
- **15 acceleration stages**, each with a beam injected via `add_species_through_plane`
- **Beam gamma ramp**: γ = 1960 + 13246 × stage_index (captures energy gain across stages)
- **PSATD solver** with multi-J algorithm (4 z-passes, 2 depositions, divE cleaning)
- **Gaussian laser pulse** (a₀=2.36) injected via laser antenna
- **Moving window** tracking the beam at velocity c

## Components

| File | Purpose |
|---|---|
| `surf_training.py` | Main framework: NN class, online normalizer, in-situ trainer, WarpX setup, execution |
| `warpx_pytorch.def` | Apptainer container definition: ROCm PyTorch + WarpX with Python bindings |
| `submit.sbatch` | SLURM job submission: 2 AMD GPU nodes, MPI + ROCm |

### `surf_training.py` sections

1. **SurrogateNN** — Feedforward neural network (6 → [800]×4 → 6) mapping initial to final particle phase-space coordinates
2. **RunningNormalizer** — Online mean/std via Welford's algorithm; no pre-computation needed
3. **InSituTrainer** — WarpX after-step callback: extracts particles, builds batches, trains, checkpoints
4. **setup_simulation()** — Full PICMI baseline: grid, plasma, beams, laser, solver
5. **main()** — Orchestrates GPU init, model creation, simulation run, final model save

## Usage

```bash
# Build container
apptainer build warpx_pytorch.sif warpx_pytorch.def

# Submit job
sbatch submit.sbatch
```

## Output

- `trained_models/surf_final_model.pt` — Final PyTorch model with full metadata
- `trained_models/surf_metadata.json` — Human-readable config, hyperparameters, training summary
- `checkpoints/` — Periodic model snapshots (every 20,000 training calls)

## References

The workflow is based on the WarpX ML surrogate training pipeline described in:

- Sandberg et al., *Hybrid beamline element ML-training for surrogates in ImpactX*, IPAC'23
- Sandberg et al., *Synthesizing Particle-In-Cell Simulations through Learning and GPU Computing*, PASC '24 (Best Paper)
- WarpX documentation: [ML dataset training workflow](https://warpx.readthedocs.io/en/24.08/usage/workflows/ml_dataset_training.html)
