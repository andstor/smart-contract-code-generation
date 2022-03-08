#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00-00:30:00         # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --nodes=1               # Allocate 1 nodes for the job
#SBATCH --cpus-per-task=12       # Allocate 40 cores
#SBATCH --gres=gpu:A100m40:2       # Allocate 4 GPUs per node
#SBATCH --job-name="GPT-J-6B-GEN"
#SBATCH --output=generate.out
#SBATCH --mail-user=andr3.storhaug@gmail.com
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of $SLURM_GPUS_ON_NODE GPUs"
echo "-----------------------------------------------------"

# Load modules
module purge
module load foss/2020b
module load Anaconda3/2020.07
module load CUDA/11.3.1

export CXX=g++

# Activate environment
source .venv/bin/activate

python ./generate.py

echo "--------------------------DONE---------------------------"