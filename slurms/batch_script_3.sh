#!/usr/bin/env bash

source /share/data/p2p/yz5880/miniforge3/bin/activate
conda activate marigold
cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
export BASE_DATA_DIR=/share/data/p2p/yz5880
export BASE_CKPT_DIR=/share/data/p2p/zhiyanw

echo "start training" 
python visualize.py --config config/train_marigold.yaml --output_dir /share/data/p2p/yz5880/Marigold_SDXL/depth_vis --resume_run /share/data/p2p/yz5880/Marigold_SDXL/depth_all_timestep/checkpoints/latest --start_num 1000 --num 500 --vis_out_dir /share/data/p2p/yz5880/Marigold_SDXL/depth_all_timestep --no_wandb
