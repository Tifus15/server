#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
#SBATCH -p amd2,amd

# Mandatory parameters
#SBATCH -J fig_threebody_2024-10-06_18-16-06
#SBATCH -a 0-0
#SBATCH -t 2-00:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=1000
#SBATCH -o ./logs/fig_threebody_2024-10-06_18-16-06/slurm_logs/%A_%a.out
#SBATCH -e ./logs/fig_threebody_2024-10-06_18-16-06/slurm_logs/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
eval "$(/mnt/beegfs/home/le/miniconda3/bin/conda shell.bash hook)"
conda activate hnn



# Program specific arguments

echo "Running scripts in parallel..."
echo "########################################################################"
            
                
python /mnt/beegfs/home/le/da/server/fig_threebody_server/main_fig.py \
		--seed $((0 + $SLURM_ARRAY_TASK_ID)) \
		--results_dir ./logs/fig_threebody_2024-10-06_18-16-06/config___configs-fig_cousins_wandb.yaml --config_file_path configs/fig_cousins_wandb.yaml --debug False --wandb_enabled True --wandb_entity andridenis --wandb_project fig_threebody --wandb_group test_group-figure-threebody --config configs/fig_cousins_wandb.yaml  &

            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
