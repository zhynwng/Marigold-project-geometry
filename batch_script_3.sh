#!/usr/bin/env bash

source /share/data/p2p/yz5880/miniforge3/bin/activate #source /root/miniforge3/bin/activate 
conda activate marigold
cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
export BASE_DATA_DIR=/share/data/p2p/yz5880
export BASE_CKPT_DIR=/share/data/p2p/zhiyanw

# env

echo "start training" 
python visualize.py --config config/train_marigold.yaml  --output_dir /share/data/p2p/yz5880/Marigold_SDXL/output/SDXL_conditional_lora  --resume_run /share/data/p2p/yz5880/Marigold_SDXL/lora_eval/SDXL_conditional_lora/train_marigold/checkpoint/latest --start_num 300 --num 100 --vis_out_dir /share/data/p2p/yz5880/visualize_lora --no_wandb
