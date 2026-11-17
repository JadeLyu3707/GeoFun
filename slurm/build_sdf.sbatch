#!/bin/bash
#SBATCH --job-name=build_sdf
#SBATCH --account=torch_pr_1317_cds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yl12783/project/phi_geometry/slurm_logs/%x_%j.out

# Hydra overrides are passed through, e.g.
#   sbatch slurm/build_sdf.sbatch builder.R=128
# (login nodes have a 3 GB memory cap, too small for the 3D grid)

module purge
export LD_LIBRARY_PATH=/scratch/yl12783/envs/mgn/lib:$LD_LIBRARY_PATH   # libEGL/libGL for open3d (missing on compute nodes)
export OMP_NUM_THREADS=8
cd /home/yl12783/phi-geometry
/scratch/yl12783/envs/mgn/bin/python build_sdf.py "$@"
