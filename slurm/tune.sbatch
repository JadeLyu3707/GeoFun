#!/bin/bash
#SBATCH --job-name=phi_tune
#SBATCH --account=torch_pr_1317_cds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yl12783/project/phi_geometry/slurm_logs/%x_%j.out

# Hydra overrides are passed through, e.g.
#   sbatch slurm/tune.sbatch data.traj=traj_00010 data.with_obstacle=False
# (login nodes have a 3 GB memory cap, too small for the 3D grid)

module purge
export LD_LIBRARY_PATH=/scratch/yl12783/envs/mgn/lib:$LD_LIBRARY_PATH   # libEGL/libGL for open3d (missing on compute nodes)
export OMP_NUM_THREADS=4
cd /home/yl12783/phi-geometry
/scratch/yl12783/envs/mgn/bin/python tune.py "$@"
