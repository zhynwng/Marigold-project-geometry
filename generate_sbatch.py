import os

# Define the number of scripts to generate
num_scripts = 20

# Template for each batch script
sbatch_template = """#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=8

cd /share/data/p2p/yz5880/cvpr2025/Marigold-project-geometry/
apptainer exec --mount type=bind,src=/scratch,dst=/scratch --mount type=bind,src=/share,dst=/share --nv /share/data/p2p/zhiyanw/new_container.sif bash slurms/batch_script_{script_number}.sh
"""

# Generate each script file
for i in range(1, num_scripts + 1):
    script_content = sbatch_template.format(script_number=i)
    script_filename = f"slurms/sbatch_{i}.sh"
    
    # Write the script content to the file
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    
    # Make the script executable
    os.chmod(script_filename, 0o755)

print(f"Generated {num_scripts} batch scripts with script numbers from 1 to {num_scripts}.")
