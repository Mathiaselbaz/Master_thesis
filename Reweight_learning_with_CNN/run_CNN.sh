#!/bin/sh
#SBATCH --time=24:00:00

#SBATCH --partition=private-dpnc-gpu

#SBATCH --gres=gpu:1,VramPerGpu:10G

#SBATCH --mem=100G

#SBATCH --output=run_CNN.out

#SBATCH --job-name='CNN_train'

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

srun singularity exec --nv -B /home,/srv $IMAGE_PATH python CNN_weight.py

echo "Job done: " `date`
