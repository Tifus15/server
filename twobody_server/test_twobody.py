import wandb
import yaml

from experiment_launcher import run_experiment, single_experiment_yaml
from twobody_local import *


def experiment(
    #######################################
    config_file_path: str = './config/twobody.yaml',

    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 41,
    results_dir: str = 'logs_osci',

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
    wandb.login()

    #######################################
    # MANDATORY

    results_dir: str = 'res_osci'

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)
    wandb.init(config=configs,project="osci_benchmark")
    twobody_loc(configs)
    wandb.finish()
    print("DONE")
























if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment) 
