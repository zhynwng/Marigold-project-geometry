import os

# Define the number of scripts to generate
num_scripts = 20

# Base content for each script
script_template = """#!/usr/bin/env bash

source /share/data/p2p/yz5880/miniforge3/bin/activate
conda activate marigold
cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
export BASE_DATA_DIR=/share/data/p2p/yz5880
export BASE_CKPT_DIR=/share/data/p2p/zhiyanw

echo "start training" 
python visualize.py --config config/train_marigold.yaml --output_dir /share/data/p2p/yz5880/Marigold_SDXL/depth_vis --resume_run /share/data/p2p/yz5880/Marigold_SDXL/depth_all_timestep/checkpoints/latest --start_num {start_num} --num 500 --vis_out_dir /share/data/p2p/yz5880/Marigold_SDXL/depth_all_timestep --no_wandb
"""

# Generate each script file
for i in range(1, num_scripts+1):
    start_num = (i-1) * 500
    script_content = script_template.format(start_num=start_num)
    script_filename = f"slurms/batch_script_{i}.sh"
    
    # Write the script content to the file
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    
    # Make the script executable
    os.chmod(script_filename, 0o755)

print(f"Generated {num_scripts} batch scripts with start_num values from 1 to {num_scripts}.")
