#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=07-00:00:00         # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --nodes=1               # Allocate 1 nodes for the job
#SBATCH --cpus-per-task=40       # Allocate 40 cores
#SBATCH --mem=1T               # Allocate 200 GB of memory per node
#SBATCH --gres=gpu:A100m40:10       # Allocate 4 GPUs per node
#SBATCH --job-name="GPT-J-6B"
#SBATCH --output=train.out
#SBATCH --mail-user=andr3.storhaug@gmail.com
#SBATCH --mail-type=ALL


cd $SLURM_SUBMIT_DIR

echo "-----------------------------------------------------"
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

# Generate hostfile
truncate -s 0 hostfile
scontrol show hostname $SLURM_NODELIST | while read line; do echo ${line}"  slots="${SLURM_GPUS_ON_NODE} >> hostfile; done

# Set up environment
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false # BF16 causes problems
export WANDB_PROJECT=smart-contracts

# Start training
deepspeed --hostfile=hostfile run_clm.py \
--deepspeed ./ds_zero2_bf16.json \
--run_name gpt-j-6B \
--model_name_or_path EleutherAI/gpt-j-6B \
--dataset_name andstor/smart_contracts \
--dataset_config_name inflated_plain_text \
--overwrite_output_dir true \
--output_dir ./out \
--report_to wandb \
--save_steps 300 \
--do_train --do_eval \
--logging_first_step --logging_steps 1 \
--num_train_epochs 2 \
--evaluation_strategy steps --eval_steps 5 \
--block_size 1024 \
--bf16 \
--gradient_accumulation_steps 16 --eval_accumulation_steps 16 \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1

echo "-------------------------- DONE ---------------------------"
