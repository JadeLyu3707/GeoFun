#!/bin/bash
#SBATCH --job-name=phi_eval
#SBATCH --account=torch_pr_1317_cds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yl12783/project/phi_geometry/slurm_logs/%x_%j.out

# Hydra overrides are passed through, e.g.
#   sbatch slurm/eval.sbatch model.sigma=0.01 model.tau=0.058
# (login nodes have a 3 GB memory cap, too small for the 3D grid)

module purge
export OMP_NUM_THREADS=4
cd /home/yl12783/phi-geometry
/scratch/yl12783/envs/mgn/bin/python eval.py "$@"
