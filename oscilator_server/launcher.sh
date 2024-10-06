#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
#SBATCH -p amd2,amd

# Mandatory parameters
#SBATCH -J osci_models_bench
#SBATCH -a 0-0
#SBATCH -t -20:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=100
#SBATCH -o /mnt/beegfs/home/le/da/osci_models_bench/slurm_logs/osci_%a.out
#SBATCH -e /mnt/beegfs/home/le/da/osci_models_bench/slurm_logs/osci_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
eval "$(/mnt/beegfs/home/le/miniconda3/bin/conda shell.bash hook)"

conda activate hnn



# Program specific arguments
echo "Running scripts in parallel..."
echo "########################################################################"
            
                
python /mnt/beegfs/home/le/da/osci/osci_local.py \
		--seed $SLURM_ARRAY_TASK_ID \
		--results_dir /mnt/beegfs/home/le/da/osci_models_bench/config___config-osci.yaml --config_file_path configs/osci.yaml --debug False --wandb_enabled True --wandb_entity andridenis --wandb_project bigbench_oscilator --wandb_group big_bench_osci --config config/osci.yaml  &

            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."

