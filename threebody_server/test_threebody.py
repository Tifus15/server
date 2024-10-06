import wandb
import yaml

from experiment_launcher import run_experiment, single_experiment_yaml
from osci_local import *


def experiment(
    #######################################
    config_file_path: str = './config/threebody.yaml',

    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 41,
    results_dir: str = 'logs_threebody',

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
    wandb.login()

    #######################################
    # MANDATORY

    results_dir: str = 'res_threebody'

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)
    wandb.init(config=configs,project="threebody_benchmark")
    osci(configs)
    wandb.finish()
    print("DONE")
























if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment) 
