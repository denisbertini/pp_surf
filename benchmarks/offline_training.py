#!/usr/bin/env python3
"""
Offline Baseline Benchmark: The "Old Way"
Loads pre-saved openPMD particle diagnostics from disk and trains the same
neural network used in the in-situ SURF framework.

Used to measure the "Control Group" for the publication:
- Total wall-clock time (Simulation + I/O + Offline Training)
- Total I/O bytes written and read
- Memory footprint during training vs. SURF's streaming approach

Usage:
    python offline_training.py --input-dir lab_particle_diags --species beam_stage_0
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- WarpX / PICMI ---
try:
    import picmistandard
    picmistandard.register_warpx()
except ImportError:
    pass

try:
    import openpmd_api as openpmd
except ImportError:
    print("[ERROR] openpmd_api is required for offline loading.")
    sys.exit(1)


# =============================================================================
# REUSE: Neural Network Class (Identical to surf_training.py)
# =============================================================================

class SurrogateNN(nn.Module):
    """
    Feedforward neural network for particle trajectory surrogate modeling.
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
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# Offline Loading & Training
# =============================================================================

def load_openpmd_data(input_dir, species, source_step, target_step):
    """
    Load initial and final particle coordinates from openPMD files.
    """
    t0 = time.time()
    
    # Load Source
    it_src = openpmd.Series(os.path.join(input_dir, "lab_particle_diags"), openpmd.AccessType.read_only)
    src_data = it_src.iterations[source_step].particle_species[species]
    
    x_src = src_data["position"][0].read().copy()
    y_src = src_data["position"][1].read().copy()
    z_src = src_data["position"][2].read().copy()
    ux_src = src_data["momentum"][0].read().copy()
    uy_src = src_data["momentum"][1].read().copy()
    uz_src = src_data["momentum"][2].read().copy()
    it_src.close()

    # Load Target
    it_tgt = openpmd.Series(os.path.join(input_dir, "lab_particle_diags"), openpmd.AccessType.read_only)
    tgt_data = it_tgt.iterations[target_step].particle_species[species]
    
    x_tgt = tgt_data["position"][0].read().copy()
    y_tgt = tgt_data["position"][1].read().copy()
    z_tgt = tgt_data["position"][2].read().copy()
    ux_tgt = tgt_data["momentum"][0].read().copy()
    uy_tgt = tgt_data["momentum"][1].read().copy()
    uz_tgt = tgt_data["momentum"][2].read().copy()
    it_tgt.close()

    io_time = time.time() - t0
    io_bytes = os.path.getsize(os.path.join(input_dir, "lab_particle_diags")) * 2 # Approximate

    source = torch.tensor(np.column_stack([x_src, y_src, z_src, ux_src, uy_src, uz_src]), dtype=torch.float32)
    target = torch.tensor(np.column_stack([x_tgt, y_tgt, z_tgt, ux_tgt, uy_tgt, uz_tgt]), dtype=torch.float32)

    return source, target, io_time, io_bytes


def normalize_data(source, target):
    """
    Batch normalization (requires full dataset in memory).
    """
    source_mean = source.mean(dim=0)
    source_std = source.std(dim=0) + 1e-20
    target_mean = target.mean(dim=0)
    target_std = target.std(dim=0) + 1e-20

    return (source - source_mean) / source_std, (target - target_mean) / target_std


def main():
    parser = argparse.ArgumentParser(description="Offline SURF Benchmark")
    parser.add_argument("--input-dir", required=True, help="Path to openPMD diagnostics")
    parser.add_argument("--species", default="beam_stage_0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    print("=" * 62)
    print("  SURF Offline Baseline (Control Group)")
    print("=" * 62)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Data (The I/O bottleneck)
    print("\n[SURF] Loading particle data from disk...")
    source, target, io_time, io_bytes = load_openpmd_data(args.input_dir, args.species, 0, 1)
    print(f"  Loaded {len(source)} particles in {io_time:.2f}s")
    print(f"  Approx I/O: {io_bytes / 1e9:.2f} GB")

    # 2. Normalize
    print("[SURF] Normalizing dataset...")
    source_norm, target_norm = normalize_data(source, target)

    # 3. Create Dataset & Dataloader
    dataset = TensorDataset(source_norm, target_norm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Initialize Model
    model = SurrogateNN(input_dim=6, output_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Train
    print(f"\n[SURF] Training for {args.epochs} epochs...")
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            pred = model(batch_x)
            loss = nn.functional.mse_loss(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.6e}")

    t_train = time.time() - t_start

    # 6. Save Results
    print("\n" + "=" * 62)
    print(f"[SURF] Benchmark Results:")
    print(f"  I/O Time             : {io_time:.2f}s ({io_bytes / 1e9:.2f} GB)")
    print(f"  Training Time        : {t_train:.2f}s")
    print(f"  Total Wall-Clock     : {io_time + t_train:.2f}s")
    
    results = {
        "io_time": io_time,
        "io_bytes": io_bytes,
        "training_time": t_train,
        "total_time": io_time + t_train,
        "particles": len(source),
        "final_loss": avg_loss,
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/offline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: benchmarks/offline_results.json")


if __name__ == "__main__":
    main()