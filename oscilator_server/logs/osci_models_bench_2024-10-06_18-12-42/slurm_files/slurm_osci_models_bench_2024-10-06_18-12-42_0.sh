#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Mandatory parameters
#SBATCH -J osci_models_bench_2024-10-06_18-12-42
#SBATCH -a 0-0
#SBATCH -t 1-00:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=1000
#SBATCH -o ./logs/osci_models_bench_2024-10-06_18-12-42/slurm_logs/%A_%a.out
#SBATCH -e ./logs/osci_models_bench_2024-10-06_18-12-42/slurm_logs/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
eval "$(/mnt/beegfs/home/le/miniconda3/bin/conda shell.bash hook)"
conda activate hnn2



# Program specific arguments

echo "Running scripts in parallel..."
echo "########################################################################"
            
                
python /mnt/beegfs/home/le/da/server/oscilator_server/test_osci.py \
		--seed $((0 + $SLURM_ARRAY_TASK_ID)) \
		--results_dir ./logs/osci_models_bench_2024-10-06_18-12-42/config___config-osci.yaml --config_file_path configs/osci.yaml --debug False --wandb_enabled True --wandb_entity andridenis --wandb_project oscilator --wandb_group bench_osci --config config/osci.yaml  &

            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
