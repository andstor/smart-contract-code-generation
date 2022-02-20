#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:15:00         # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --nodes=1               # Allocate 1 nodes for the job
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100m40:4    # Allocate 4 GPUs per node
#SBATCH --job-name="GPT-NeoX-20B"
#SBATCH --output=gpt-neox.out
#SBATCH --mail-user=andr3.storhaug@gmail.com
#SBATCH --mail-type=ALL

MODEL_DIR="${SLURM_SUBMIT_DIR}/models/gpt-neox"
CONFIG_DIR="${SLURM_SUBMIT_DIR}/.hpc/gpt-neox/configs/"

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "-----------------------------------------------------"

module purge
module load foss/2020b
module load Anaconda3/2020.07
module load CUDA/11.3.1

export CXX=g++

cd $MODEL_DIR

source .venv/bin/activate

python ./deepy.py generate.py -d configs ${CONFIG_DIR}/20B.yml ${CONFIG_DIR}/text_generation.yml