import yaml
from main_pend import *



if __name__ == "__main__":
    
    with open("configs/pend.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)
    folder_name1 = "a43"
    full_server4(configs,folder_name1)

