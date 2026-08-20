#!/usr/bin/env python3
"""
SURF: Surrogate Using Real-time Flow
In-situ surrogate modeling framework for WarpX laser-plasma accelerator simulations.

Trains a PyTorch neural network surrogate during WarpX simulation execution,
directly from live particle data in memory, completely bypassing disk I/O.

Usage:
    python surf_training.py

Build container:
    apptainer build warpx_pytorch.sif warpx_pytorch.def

Submit job:
    sbatch submit.sbatch
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

import os
import sys
import time
import json
import traceback
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# --- WarpX / PICMI ---
try:
    import picmistandard
    picmistandard.register_warpx()
except ImportError as e:
    print(f"[SURF] Error: picmistandard not available: {e}")
    sys.exit(1)

try:
    import picmi
except ImportError as e:
    print(f"[SURF] Error: picmi not available: {e}")
    sys.exit(1)

try:
    from pywarpx import warpx
    from pywarpx import callbacks
except ImportError as e:
    print(f"[SURF] Error: pywarpx not available. Ensure WarpX is built with Python bindings.")
    sys.exit(1)

try:
    import openpmd_api as openpmd
except ImportError:
    openpmd = None

# =============================================================================
# SECTION 2: NEURAL NETWORK CLASS
# =============================================================================

class SurrogateNN(nn.Module):
    """
    Feedforward neural network for particle trajectory surrogate modeling.

    Maps initial particle phase-space coordinates (x, y, z, px, py, pz) to
    final coordinates after acceleration through the plasma stage.

    Parameters:
        input_dim: Input feature count (default 6)
        output_dim: Output feature count (default 6)
        hidden_layers: List of hidden layer sizes (default [800, 800, 800, 800])
        activation: Activation function ('relu', 'tanh', 'prelu', 'sigmoid')
    """

    def __init__(self, input_dim=6, output_dim=6,
                 hidden_layers=None, activation='relu'):
        super(SurrogateNN, self).__init__()

        if hidden_layers is None:
            hidden_layers = [800, 800, 800, 800]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = list(hidden_layers)
        self.activation_name = activation

        layers = []
        in_features = input_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'prelu':
                layers.append(nn.PReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

    def get_config(self):
        """Return serializable architecture configuration dict."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation_name,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


# =============================================================================
# SECTION 3: ONLINE NORMALIZER (WELFORD'S ALGORITHM)
# =============================================================================

class RunningNormalizer:
    """
    Online normalizer using Welford's algorithm for numerically stable
    running mean and variance over streaming particle data.

    Eliminates the need for a pre-computation phase, enabling true
    single-pass in-situ training.
    """

    def __init__(self, dim):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch):
        """
        Update running statistics with a new batch.

        Parameters:
            batch: numpy array of shape (N, dim)
        """
        batch = np.asarray(batch, dtype=np.float64)
        batch_n = batch.shape[0]
        if batch_n == 0:
            return

        batch_mean = batch.mean(axis=0)

        self.n += batch_n
        delta = batch_mean - self.mean
        delta_n = delta * batch_n / self.n
        term1 = self.M2 * batch_n / self.n

        self.M2 += term1
        self.mean += delta_n * (batch_n - 1) / self.n
        self.M2 += batch.var(axis=0, ddof=1) * (batch_n - 1) * batch_n / self.n

    @property
    def std(self):
        """Current running standard deviation."""
        if self.n < 2:
            return np.ones(self.dim, dtype=np.float64)
        variance = self.M2 / (self.n - 1)
        return np.sqrt(np.maximum(variance, 1e-20))

    def normalize(self, data):
        """Normalize data using current running mean/std."""
        std = self.std
        return (data - self.mean) / std

    def unnormalize(self, data):
        """Inverse transform."""
        return data * self.std + self.mean

    def get_stats(self):
        """Return statistics as serializable dict."""
        return {
            "count": int(self.n),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


# =============================================================================
# SECTION 4: IN-SITU TRAINER
# =============================================================================

class InSituTrainer:
    """
    Encapsulates all in-situ training state and logic.

    Registered as a WarpX after-step callback. At every N simulation steps,
    extracts live particle data from WarpX memory, normalizes it using
    running statistics, performs a training step on the surrogate model,
    and logs metrics.
    """

    def __init__(self, model, optimizer, input_normalizer, output_normalizer,
                  train_interval=1000, min_particles=100, max_batch_size=5000,
                  particle_name="beam_stage_0", checkpoint_interval=20000,
                  z_filter_min=None, z_filter_max=None, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.train_interval = train_interval
        self.min_particles = min_particles
        self.max_batch_size = max_batch_size
        self.particle_name = particle_name
        self.checkpoint_interval = checkpoint_interval
        self.z_filter_min = z_filter_min
        self.z_filter_max = z_filter_max
        self.device = device

        self.loss_history = []
        self.step_count = 0
        self.train_count = 0
        self.start_time = time.time()
        self.total_train_time = 0.0
        self.rng = np.random.default_rng(42)

        # Buffer for previous-step particle data (input->target pairs)
        self._prev_input = None

        print(f"[SURF] Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Train interval: every {train_interval} steps")
        print(f"  Particle species: {particle_name}")
        print(f"  Max batch size: {max_batch_size}")
        print(f"  Min particles: {min_particles}")
        if z_filter_min is not None or z_filter_max is not None:
            print(f"  Z-filter: [{z_filter_min}, {z_filter_max}]")

    # -----------------------------------------------------------------------
    # Particle data extraction
    # -----------------------------------------------------------------------

    def _extract_particle_data(self, sim):
        """
        Extract live particle data from the WarpX simulation object.

        Tries multiple WarpX Python API variants to extract position and
        momentum arrays.  Returns numpy arrays (x, y, z, ux, uy, uz) or
        (None, ...) on failure.
        """

        def _try_extract():
            """Return (pos, mom) or raise."""
            # ----- Variant 1: sim.particles.get() ------
            try:
                ptc = sim.particles.get(self.particle_name)
                if ptc is not None:
                    pos = ptc.get_position()
                    mom = ptc.get_momentum()
                    if pos is not None and mom is not None:
                        return pos, mom
            except Exception:
                pass

            # ----- Variant 2: warpx.particles.get() -----
            try:
                ptc = warpx.particles.get(self.particle_name)
                if ptc is not None:
                    pos = ptc.get_position()
                    mom = ptc.get_momentum()
                    if pos is not None and mom is not None:
                        return pos, mom
            except Exception:
                pass

            # ----- Variant 3: warpx.get_particle_container() -----
            try:
                ptc = warpx.get_particle_container(self.particle_name)
                if ptc is not None:
                    pos = ptc.get_particle_data("position")
                    mom = ptc.get_particle_data("momentum")
                    if pos is not None and mom is not None:
                        return pos, mom
            except Exception:
                pass

            # ----- Variant 4: sim.particles.get() + get_particle_data() -----
            try:
                ptc = sim.particles.get(self.particle_name)
                if ptc is not None:
                    pos = ptc.get_particle_data("position")
                    mom = ptc.get_particle_data("momentum")
                    if pos is not None and mom is not None:
                        return pos, mom
            except Exception:
                pass

            raise RuntimeError(
                f"Could not extract particle data for '{self.particle_name}' "
                f"(tried 4 API variants)"
            )

        try:
            pos, mom = _try_extract()

            x = pos[:, 0].copy()
            y = pos[:, 1].copy()
            z = pos[:, 2].copy()
            ux = mom[:, 0].copy()
            uy = mom[:, 1].copy()
            uz = mom[:, 2].copy()

            return x, y, z, ux, uy, uz

        except Exception as e:
            print(f"[SURF] Failed to extract particle data: {e}")
            return None, None, None, None, None, None

    def _filter_particles(self, x, y, z, ux, uy, uz):
        """Filter particles based on z position bounds."""
        if self.z_filter_min is None and self.z_filter_max is None:
            return x, y, z, ux, uy, uz

        mask = np.ones(len(z), dtype=bool)
        if self.z_filter_min is not None:
            mask &= z >= self.z_filter_min
        if self.z_filter_max is not None:
            mask &= z <= self.z_filter_max

        return x[mask], y[mask], z[mask], ux[mask], uy[mask], uz[mask]

    # -----------------------------------------------------------------------
    # Batch construction and normalization
    # -----------------------------------------------------------------------

    def _build_batch(self, x, y, z, ux, uy, uz):
        """
        Build a normalized mini-batch from particle data.

        Constructs input = previous-step state, target = current-step state.
        Samples max_batch_size particles when more are available.
        Returns (input_tensor, target_tensor) or (None, None) if insufficient data.
        """
        n = len(x)
        if n < self.min_particles:
            return None, None

        # Random subsampling if too many particles
        if n > self.max_batch_size:
            indices = self.rng.choice(n, self.max_batch_size, replace=False)
            x = x[indices]
            y = y[indices]
            z = z[indices]
            ux = ux[indices]
            uy = uy[indices]
            uz = uz[indices]
            n = len(x)

        current_data = np.column_stack([x, y, z, ux, uy, uz]).astype(np.float32)

        # Need previous-step data to form (input -> target) pairs
        if self._prev_input is None:
            self._prev_input = current_data
            return None, None

        input_data = self._prev_input.astype(np.float64)
        target_data = current_data.astype(np.float64)

        # Update online statistics
        self.input_normalizer.update(input_data)
        self.output_normalizer.update(target_data)

        # Normalize
        norm_input = self.input_normalizer.normalize(input_data)
        norm_target = self.output_normalizer.normalize(target_data)

        # Store current state for next interval
        self._prev_input = current_data

        # Transfer to GPU
        input_tensor = torch.tensor(
            norm_input, dtype=torch.float32, device=self.device
        )
        target_tensor = torch.tensor(
            norm_target, dtype=torch.float32, device=self.device
        )

        return input_tensor, target_tensor

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def _training_step(self, input_tensor, target_tensor):
        """Single training step: forward pass, MSE loss, backward, optimize."""
        self.model.train()

        prediction = self.model(input_tensor)
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    # -----------------------------------------------------------------------
    # Checkpoint / save
    # -----------------------------------------------------------------------

    def _save_checkpoint(self):
        """Save model checkpoint with full metadata (rank 0 only)."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        filepath = os.path.join(
            checkpoint_dir,
            f"surf_step{self.step_count}_train{self.train_count}.pt"
        )

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "step_count": self.step_count,
            "train_count": self.train_count,
            "total_train_time": self.total_train_time,
            "input_stats": self.input_normalizer.get_stats(),
            "output_stats": self.output_normalizer.get_stats(),
            "model_config": self.model.get_config(),
            "timestamp": datetime.now().isoformat(),
        }, filepath)

        # Save loss history as JSON for external monitoring
        with open(os.path.join(checkpoint_dir, "loss_history.json"), "w") as f:
            json.dump({
                "steps": list(range(len(self.loss_history))),
                "losses": self.loss_history,
                "train_count": self.train_count,
            }, f)

        print(f"[SURF] Checkpoint saved: {filepath}")

    # -----------------------------------------------------------------------
    # WarpX callback
    # -----------------------------------------------------------------------

    def callback(self, sim):
        """
        Main callback function installed with callbacks.installafterstep().

        Executes after every simulation step. Trains only at intervals and
        only on MPI rank 0.
        """
        self.step_count += 1

        # Only train at specified intervals
        if self.step_count % self.train_interval != 0:
            return

        # Determine MPI rank
        try:
            mpi_rank = warpx.ParallelDescriptor.Rank()
        except Exception:
            mpi_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))

        if mpi_rank != 0:
            return

        t0 = time.time()
        loss_value = None

        try:
            # 1. Extract live particle data from WarpX memory
            x, y, z, ux, uy, uz = self._extract_particle_data(sim)

            if x is None or len(x) == 0:
                return

            # 2. Filter particles (e.g., by z position)
            x, y, z, ux, uy, uz = self._filter_particles(x, y, z, ux, uy, uz)

            if len(x) < self.min_particles:
                return

            # 3. Build normalized training batch
            input_tensor, target_tensor = self._build_batch(x, y, z, ux, uy, uz)

            if input_tensor is None or target_tensor is None:
                return

            # 4. Training step
            loss_value = self._training_step(input_tensor, target_tensor)
            self.loss_history.append(loss_value)
            self.train_count += 1

        except Exception as e:
            print(f"[SURF] Training error at step {self.step_count}: {e}")
            traceback.print_exc()
            self.total_train_time += time.time() - t0
            return

        elapsed = time.time() - t0
        self.total_train_time += elapsed

        # Log every 10 training calls or on first call
        if loss_value is not None and (self.train_count % 10 == 0 or self.train_count == 1):
            wall_min = (time.time() - self.start_time) / 60.0
            avg_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"[SURF] Step {self.step_count:8d} | "
                f"Train #{self.train_count:5d} | "
                f"Loss: {loss_value:.6e} | "
                f"N: {len(x):6d} | "
                f"dt: {elapsed:.3f}s | "
                f"Wall: {wall_min:6.1f}min | "
                f"LR: {avg_lr:.0e}"
            )

        # Periodic checkpoint
        if self.train_count % self.checkpoint_interval == 0:
            self._save_checkpoint()

        # Free GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Final model save
    # -----------------------------------------------------------------------

    def save_final_model(self):
        """Save the final trained model with complete metadata."""
        output_dir = "trained_models"
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "surf_final_model.pt")
        meta_path = os.path.join(output_dir, "surf_metadata.json")

        wall_time = time.time() - self.start_time

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "step_count": self.step_count,
            "train_count": self.train_count,
            "total_train_time": self.total_train_time,
            "wall_time": wall_time,
            "input_stats": self.input_normalizer.get_stats(),
            "output_stats": self.output_normalizer.get_stats(),
            "model_config": self.model.get_config(),
            "hyperparameters": {
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "train_interval": self.train_interval,
                "max_batch_size": self.max_batch_size,
                "min_particles": self.min_particles,
                "particle_name": self.particle_name,
                "device": self.device,
                "hidden_layers": self.model.hidden_layers,
                "activation": self.model.activation_name,
            },
            "simulation_params": {
                "nx": 128, "ny": 128, "nz": 35328,
                "n0": 1.7e23, "L_plasma_bulk": 0.28,
                "n_stages": 15, "particles_per_stage": 1e6,
            },
        }

        torch.save(checkpoint, model_path)

        json_meta = {
            "model_config": self.model.get_config(),
            "hyperparameters": checkpoint["hyperparameters"],
            "simulation_params": checkpoint["simulation_params"],
            "training_summary": {
                "total_steps": self.step_count,
                "total_training_calls": self.train_count,
                "total_train_time_sec": round(self.total_train_time, 2),
                "wall_time_sec": round(wall_time, 2),
                "final_loss": self.loss_history[-1] if self.loss_history else None,
                "initial_loss": self.loss_history[0] if self.loss_history else None,
            },
            "normalization_stats": {
                "input": self.input_normalizer.get_stats(),
                "output": self.output_normalizer.get_stats(),
            },
        }

        with open(meta_path, "w") as f:
            json.dump(json_meta, f, indent=2)

        print(f"\n{'=' * 62}")
        print(f"[SURF] Training complete!")
        print(f"[SURF]   Total simulation steps : {self.step_count}")
        print(f"[SURF]   Training calls         : {self.train_count}")
        print(f"[SURF]   Total train time       : {self.total_train_time:.1f}s")
        print(f"[SURF]   Wall time              : {wall_time/60:.1f} min")
        if self.loss_history:
            print(f"[SURF]   Initial loss           : {self.loss_history[0]:.6e}")
            print(f"[SURF]   Final loss             : {self.loss_history[-1]:.6e}")
        print(f"[SURF]   Model saved            : {model_path}")
        print(f"[SURF]   Metadata saved         : {meta_path}")
        print(f"{'=' * 62}\n")


# =============================================================================
# SECTION 5: SIMULATION SETUP
# =============================================================================

def setup_simulation():
    """
    Configure the WarpX laser-plasma accelerator simulation using the full
    PICMI baseline from the WarpX ML dataset training workflow.

    Critical physics components:
        - Boosted frame (gamma_boost=60)
        - Cartesian 3D grid with moving window
        - Parabolic plasma density profile with cosine ramps
        - 15 acceleration stages with per-stage beam gamma ramp-up
        - PSATD solver with multi-J algorithm (4 passes, 2 depositions)
        - Gaussian laser with antenna injection
        - Beam injection through plane with pseudo-random layout
        - Stage spacing with focusing lens parameters
    """

    import math

    # --- Physical constants ---
    c = picmi.constants.c
    q_e = picmi.constants.q_e
    m_e = picmi.constants.m_e
    ep0 = picmi.constants.ep0

    # --- Grid parameters ---
    nx = 128
    ny = 128
    nz = 35328

    # --- Computational domain ---
    rmax = 128e-6
    zmin = -180e-6
    zmax = 0.0

    # --- Boosted frame ---
    gamma_boost = 60.0
    # --- Plasma parameters ---
    plasma_rlim = 100.0e-6
    n0 = 1.7e23
    L_plasma_bulk = 0.28
    L_ramp = 1.0e-9
    L_stage = L_plasma_bulk + 2 * L_ramp

    # --- Focusing / stage spacing ---
    N_stage = 15
    lens_focal_length = 0.015
    lens_width = 0.003
    stage_spacing = L_plasma_bulk + 2 * lens_focal_length

    # --- Beam parameters ---
    N_beam_particles = int(1e6)
    beam_charge = -10.0e-15
    beam_centroid_z = -107.0e-6
    beam_rms_z = 2.0e-6
    beam_gammas = [1960 + 13246 * i for i in range(N_stage)]

    # --- Laser parameters ---
    antenna_z = -1e-9
    profile_t_peak = 1.46764864e-13

    # --- MPI process distribution ---
    num_procs = [1, 1, 256]

    # =================================================================
    # Grid with moving window
    # =================================================================
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[nx, ny, nz],
        guard_cells=[11, 11, 12],
        lower_bound=[-rmax, -rmax, zmin],
        upper_bound=[rmax, rmax, zmax],
        lower_boundary_conditions=['periodic', 'periodic', 'damped'],
        upper_boundary_conditions=['periodic', 'periodic', 'damped'],
        lower_boundary_conditions_particles=['periodic', 'periodic', 'absorbing'],
        upper_boundary_conditions_particles=['periodic', 'periodic', 'absorbing'],
        moving_window_velocity=[0.0, 0.0, c],
        warpx_max_grid_size=256,
        warpx_blocking_factor=32,
    )

    # =================================================================
    # Plasma: parabolic density profile with cosine ramps per stage
    # =================================================================
    kp = q_e / c * math.sqrt(n0 / (m_e * ep0))
    Rc = 40.0e-6
    pi = math.pi

    def get_stage_plasma(stage_idx, stage_zmin, stage_zmax,
                         Lplus=L_ramp, Lp=L_plasma_bulk, Lminus=L_ramp):
        """Parabolic transverse density profile with ramp-up and ramp-down."""
        density_expr = (
            f'n0*(1.+4.*(x**2+y**2)/(kp**2*Rc**4))'
            f'*(0.5*(1.-cos(pi*(z-{stage_zmin})/Lplus)))*((z-{stage_zmin})<Lplus)'
            f'+n0*(1.+4.*(x**2+y**2)/(kp**2*Rc**4))'
            f'*((z-{stage_zmin})>=Lplus)*((z-{stage_zmin})<(Lplus+Lp))'
            f'+n0*(1.+4.*(x**2+y**2)/(kp**2*Rc**4))'
            f'*(0.5*(1.+cos(pi*((z-{stage_zmin})-Lplus-Lp)/Lminus)))'
            f'*((z-{stage_zmin})>=(Lplus+Lp))*((z-{stage_zmin})<(Lplus+Lp+Lminus))'
        )

        dist = picmi.AnalyticDistribution(
            density_expression=density_expr,
            pi=pi,
            n0=n0,
            kp=kp,
            Rc=Rc,
            Lplus=Lplus,
            Lp=Lp,
            Lminus=Lminus,
            lower_bound=[-plasma_rlim, -plasma_rlim, stage_zmin],
            upper_bound=[plasma_rlim, plasma_rlim, stage_zmax],
            fill_in=True,
        )

        electrons = picmi.Species(
            particle_type='electron',
            name=f'electrons{stage_idx}',
            initial_distribution=dist,
        )
        ions = picmi.Species(
            particle_type='proton',
            name=f'ions{stage_idx}',
            initial_distribution=dist,
        )
        return electrons, ions

    species_list = []
    for i_stage in range(1):
        zmin_s = i_stage * stage_spacing
        zmax_s = zmin_s + L_stage
        electrons, ions = get_stage_plasma(i_stage + 1, zmin_s, zmax_s)
        species_list.append(electrons)
        species_list.append(ions)

    # =================================================================
    # Beam: GaussianBunch per stage, injected through plane
    # =================================================================
    beams = []
    for i_stage in range(N_stage):
        beam_gamma = beam_gammas[i_stage]
        sigma_gamma = 0.06 * beam_gamma

        dist = picmi.GaussianBunchDistribution(
            n_physical_particles=abs(beam_charge) / q_e,
            rms_bunch_size=[2.0e-6, 2.0e-6, beam_rms_z],
            rms_velocity=[8 * c, 8 * c, sigma_gamma * c],
            centroid_position=[0.0, 0.0, beam_centroid_z],
            centroid_velocity=[0.0, 0.0, beam_gamma * c],
        )

        beam = picmi.Species(
            particle_type='electron',
            name=f'beam_stage_{i_stage}',
            initial_distribution=dist,
        )
        beams.append(beam)

    # =================================================================
    # Laser: GaussianLaser with antenna injection
    # =================================================================
    def get_laser(az, t_peak, fill_in=True):
        focal_distance = 0.0
        laser = picmi.GaussianLaser(
            wavelength=0.8e-6,
            waist=36e-6,
            duration=7.33841e-14,
            focal_position=[0.0, 0.0, focal_distance + az],
            centroid_position=[0.0, 0.0, az - c * t_peak],
            propagation_direction=[0.0, 0.0, 1.0],
            polarization_direction=[0.0, 1.0, 0.0],
            a0=2.36,
            fill_in=fill_in,
        )
        antenna = picmi.LaserAntenna(
            position=[0.0, 0.0, az],
            normal_vector=[0.0, 0.0, 1.0],
        )
        return laser, antenna

    lasers = [get_laser(antenna_z, profile_t_peak, fill_in=False)]

    # =================================================================
    # PSATD solver: multi-J algorithm (4 z-passes, 2 depositions)
    # =================================================================
    n_pass_z = 4
    smoother = picmi.BinomialSmoother(n_pass=[1, 1, n_pass_z])
    stencil_order = [8, 8, 16]
    grid_type = 'hybrid'

    solver = picmi.ElectromagneticSolver(
        grid=grid,
        method='PSATD',
        cfl=0.9999,
        source_smoother=smoother,
        stencil_order=stencil_order,
        galilean_velocity=None,
        warpx_psatd_update_with_rho=True,
        warpx_current_correction=False,
        divE_cleaning=True,
        warpx_psatd_J_in_time='linear',
    )

    # =================================================================
    # Reduced diagnostic for beam monitoring (lightweight)
    # =================================================================
    beamrel_red_diag = picmi.ReducedDiagnostic(
        diag_type='BeamRelevant',
        name='beamrel',
        species=beams[0],
        period=1,
    )

    # =================================================================
    # Simulation object
    # =================================================================
    sim = picmi.Simulation(
        solver=solver,
        warpx_numprocs=num_procs,
        warpx_compute_max_step_from_btd=True,
        verbose=2,
        particle_shape='cubic',
        gamma_boost=gamma_boost,
        warpx_charge_deposition_algo='standard',
        warpx_current_deposition_algo='direct',
        warpx_field_gathering_algo='momentum-conserving',
        warpx_particle_pusher_algo='vay',
        warpx_amrex_the_arena_is_managed=False,
        warpx_amrex_use_gpu_aware_mpi=True,
        warpx_do_multi_J=True,
        warpx_do_multi_J_n_depositions=2,
        warpx_grid_type=grid_type,
        warpx_field_centering_order=[16, 16, 16],
        warpx_current_centering_order=[16, 16, 16],
    )

    # Add plasma species (gridded layout)
    for sp in species_list:
        sim.add_species(
            sp,
            layout=picmi.GriddedLayout(
                grid=grid,
                n_macroparticle_per_cell=[2, 2, 2],
            ),
        )

    # Add beam species through plane (pseudo-random layout)
    for i_stage in range(N_stage):
        sim.add_species_through_plane(
            species=beams[i_stage],
            layout=picmi.PseudoRandomLayout(
                grid=grid,
                n_macroparticles=N_beam_particles,
            ),
            injection_plane_position=0.0,
            injection_plane_normal_vector=[0.0, 0.0, 1.0],
        )

    # Add laser with antenna
    laser, laser_antenna = lasers[0]
    sim.add_laser(laser, injection_method=laser_antenna)

    # Add diagnostics
    sim.add_diagnostic(beamrel_red_diag)

    sim.initialize_inputs()

    return sim


# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

def main():
    """
    Entry point: verify GPU, initialize model and optimizer, set up
    WarpX simulation, register in-situ training callback, run simulation,
    and save the final trained model.
    """

    print()
    print("=" * 62)
    print("  SURF: Surrogate Using Real-time Flow")
    print("  In-situ Neural Surrogate Training for WarpX")
    print("=" * 62)
    print()

    # --- GPU verification ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[SURF] GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
        print(f"[SURF] PyTorch: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[SURF] WARNING: CUDA not detected, falling back to CPU.")
        print("[SURF] For production use, ensure NVIDIA GPU access is configured.")

    # --- Initialize surrogate neural network ---
    print("\n[SURF] Initializing surrogate model...")
    hidden_layers = [800, 800, 800, 800]
    model = SurrogateNN(
        input_dim=6,
        output_dim=6,
        hidden_layers=hidden_layers,
        activation='relu',
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SURF] Architecture: 6 -> {hidden_layers} -> 6")
    print(f"[SURF] Parameters: {total_params:,}")

    # --- Optimizer ---
    learning_rate = 1e-4
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    # --- Online normalizers ---
    input_normalizer = RunningNormalizer(dim=6)
    output_normalizer = RunningNormalizer(dim=6)

    # --- WarpX simulation setup ---
    print("\n[SURF] Setting up WarpX simulation...")
    sim = setup_simulation()
    print("[SURF] Simulation initialized.")

    # --- In-situ trainer ---
    trainer = InSituTrainer(
        model=model,
        optimizer=optimizer,
        input_normalizer=input_normalizer,
        output_normalizer=output_normalizer,
        train_interval=1000,
        min_particles=100,
        max_batch_size=5000,
        particle_name="beam_stage_0",
        checkpoint_interval=20000,
        z_filter_min=None,
        z_filter_max=None,
        device=device,
    )

    # --- Register WarpX callback ---
    print("[SURF] Registering in-situ training callback...")
    callbacks.installafterstep(trainer.callback)

    # --- Run simulation ---
    print("\n[SURF] Starting simulation with in-situ training...\n")
    sim.step()

    # --- Save final model ---
    trainer.save_final_model()


if __name__ == "__main__":
    main()
