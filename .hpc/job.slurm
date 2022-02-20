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

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "-----------------------------------------------------"

module load foss/2020b
module load Anaconda3/2020.07
module load CUDA/11.3.1

export CXX=g++

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

#pip install torch
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements/requirements.txt

python ./megatron/fused_kernels/setup.py install

./deepy.py generate.py ./configs/20B.yml

# Run with sbatch job.slurm