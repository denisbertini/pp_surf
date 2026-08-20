#!/usr/bin/env bash
# =============================================================================
# SURF: Surrogate Using Real-time Flow
# SLURM job submission script (ROCm / AMD MI100)
#
# Architecture:
#   - 2+ AMD GPU nodes: WarpX MPI across MI100 GPUs (ROCm)
#   - Rank 0: extracts particle data -> PyTorch trains on MI100 GPU (ROCm)
#   - Both WarpX and PyTorch use the same MI100 GPUs via ROCm
#
# Usage:
#   sbatch submit.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f surf_<JOBID>.out
#
# =============================================================================

#SBATCH --job-name=surf_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:mi100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=amd_gpu
#SBATCH --output=surf_%j.out
#SBATCH --error=surf_%j.err
#SBATCH --requeue

set -euo pipefail

echo "=========================================="
echo " SURF Job: ${SLURM_JOB_ID}"
echo " Started: $(date)"
echo " Nodes:   ${SLURM_JOB_NUM_NODES}"
echo " MPI:     ${SLURM_NTASKS} ranks (${SLURM_NTASKS_PER_NODE}/node)"
echo " GPUs:    AMD MI100 (ROCm)"
echo "=========================================="

module purge
module load rocm
module load apptainer

CONTAINER="/home/denis/work/pp/surf/warpx_pytorch.sif"
SCRIPT="/home/denis/work/pp/surf/surf_training.py"
WORK_DIR="/home/denis/work/pp/surf"

if [ ! -f "${CONTAINER}" ]; then
    echo "[ERROR] Container not found: ${CONTAINER}"
    echo "  Build: apptainer build ${CONTAINER} warpx_pytorch.def"
    exit 1
fi

if [ ! -f "${SCRIPT}" ]; then
    echo "[ERROR] Script not found: ${SCRIPT}"
    exit 1
fi

# ROCm environment
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# MPI
export OMPI_MCA_btl_tcp_if_include=eth0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${WORK_DIR}/checkpoints" "${WORK_DIR}/trained_models"

echo ""
echo "[SURF] Container: ${CONTAINER}"
echo "[SURF] Nodes: ${SLURM_JOB_NUM_NODES}, MPI ranks: ${SLURM_NTASKS}"
echo ""

srun \
    --nodes=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    --gres=gpu:mi100:1 \
    apptainer exec --rocm \
        --bind "${WORK_DIR}:${WORK_DIR}" \
        --cwd "${WORK_DIR}" \
        "${CONTAINER}" \
        mpirun -np ${SLURM_NTASKS} \
        python3 "${SCRIPT}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo " SURF Job ${SLURM_JOB_ID} finished (exit: ${EXIT_CODE})"
echo " Ended: $(date)"
echo "=========================================="

if [ -f "${WORK_DIR}/trained_models/surf_final_model.pt" ]; then
    du -h "${WORK_DIR}/trained_models/surf_final_model.pt"
    echo "[OK] Model saved."
else
    echo "[WARN] No model found."
fi

exit ${EXIT_CODE}
