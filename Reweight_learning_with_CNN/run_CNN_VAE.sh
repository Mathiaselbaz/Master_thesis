#!/bin/env bash

#SBATCH --time=12:00:00

#SBATCH --partition=shared-gpu

#SBATCH --gres=gpu:6,VramPerGpu:10G

#SBATCH --mem=100G

#SBATCH --output=CNN_VAE.out

#SBATCH --job-name='VAE_CNN'

# You will need to set this to whatever directory you are working from

#SBATCH --chdir=/home/users/e/elbazma1/nflows/nflows

export IMAGE_PATH="week4.simg"

export XDG_RUNTIME_DIR=""

echo "RUNDIR=/home/student/"



echo "Starting job: " `date`

hostname


module load GCC/9.3.0 Singularity/3.7.3-Go-1.14



# launch Jupyter notebook

srun singularity exec --nv -B /home,/srv $IMAGE_PATH python CNN_VAE_new.py $1 $2 $3 $4 $5

echo "Job done: " `date`
