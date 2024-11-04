#!/usr/bin/env bash

source /share/data/p2p/yz5880/miniforge3/bin/activate #source /root/miniforge3/bin/activate 
conda activate marigold
cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
export BASE_DATA_DIR=/scratch/yz5880/Projective-Geometry-1024/Indoor_Real_1024
export BASE_CKPT_DIR=/share/data/p2p/zhiyanw

env

echo "start training" 
python train.py --config config/train_marigold.yaml --output_dir /share/data/p2p/yz5880/Marigold_SDXL/conditional_prompt/
