#!/bin/bash

# Source the conda environment
. /home/ptreyer/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate MedMNIST

# Get the GPU with the least usage based on utilization, memory used, and index
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
  | sort -t',' -k2,2n -k3,3n -k1,1rn \
  | awk -F',' '{print $1}' \
  | head -n 1 \
  | { read GPU; echo "GPU used: $GPU"; CUDA_VISIBLE_DEVICES=$GPU python Project/Plot/main.py; }

# Optionally, you can also uncomment the following lines if you want to set OMP_NUM_THREADS
# OMP_NUM_THREADS=1 sh run_main.txt 0=, 1=1, 2=, 4=4, 5=5
