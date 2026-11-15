#!/bin/bash
#SBATCH --job-name=h5_convert
#SBATCH --account=torch_pr_1317_cds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yl12783/project/phi_geometry/slurm_logs/%x_%j.out

module purge
cd /home/yl12783/phi-geometry
/scratch/yl12783/envs/mgn/bin/python scripts/tfrecord_to_h5.py "$@"
