#!/bin/env bash

#SBATCH --time=02:00:00

#SBATCH --partition=shared-cpu

#SBATCH --mem=100G

#SBATCH --output=run_CNN_test.out

#SBATCH --job-name='weight_calc'

# You will need to set this to whatever directory you are working from

#SBATCH --chdir=/home/users/e/elbazma1/nflows

export IMAGE_PATH="week4.simg"

export XDG_RUNTIME_DIR=""

echo "RUNDIR=/home/student/"



echo "Starting job: " `date`

hostname



module load GCCcore/8.2.0

module load Singularity/3.4.0-Go-1.12



# launch Jupyter notebook

srun singularity exec -B /home,/srv $IMAGE_PATH python CNN_test.py

echo "Job done: " `date`
