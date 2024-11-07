#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --export=ALL

cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
apptainer exec --mount type=bind,src=/scratch,dst=/scratch --mount type=bind,src=/share,dst=/share --nv /share/data/p2p/zhiyanw/container.sif bash batch_script.sh