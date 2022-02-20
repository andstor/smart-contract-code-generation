#!/bin/bash

scriptpath="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
model_path="${scriptpath}/../../models/gpt-neox"

cd $model_path

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