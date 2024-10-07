#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
#SBATCH -p amd2,amd

# Mandatory parameters
#SBATCH -J diverse_threebody_2024-10-07_01-38-21
#SBATCH -a 0-0
#SBATCH -t 2-00:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=1000
#SBATCH -o ./logs/diverse_threebody_2024-10-07_01-38-21/slurm_logs/%A_%a.out
#SBATCH -e ./logs/diverse_threebody_2024-10-07_01-38-21/slurm_logs/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
eval "$(/mnt/beegfs/home/le/miniconda3/bin/conda shell.bash hook)"
conda activate hnn2



# Program specific arguments

echo "Running scripts in parallel..."
echo "########################################################################"
            
                
python /mnt/beegfs/home/le/da/server/div_threebody_server/main_div.py \
		--seed $((0 + $SLURM_ARRAY_TASK_ID)) \
		--results_dir ./logs/diverse_threebody_2024-10-07_01-38-21/config___configs-diverse_wandb.yaml --config_file_path configs/diverse_wandb.yaml --debug False --wandb_enabled True --wandb_entity andridenis --wandb_project diverse_threebody --wandb_group test_group-diverse-threebody --config configs/diverse_wandb.yaml  &

            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
