#!/usr/bin/env bash

source /share/data/p2p/yz5880/miniforge3/bin/activate #source /root/miniforge3/bin/activate 
conda activate marigold
cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
export BASE_DATA_DIR=/scratch/yz5880
export BASE_CKPT_DIR=/share/data/p2p/zhiyanw

# env

echo "start training" 
# python train.py --config config/train_marigold.yaml  --output_dir /share/data/p2p/yz5880/SDXL_depth/
python visualize.py --config config/train_marigold.yaml  --output_dir /share/data/p2p/zhiyanw/SDXL_depth/all_pf_timestep  --resume_run /share/data/p2p/zhiyanw/SDXL_depth/all_pf_timestep/train_marigold/checkpoint/latest  --start_num 1 --num 100 --vis_out_dir /share/data/p2p/zhiyanw/SDXL_depth_visualzation/all_pf_timestep --no_wandb