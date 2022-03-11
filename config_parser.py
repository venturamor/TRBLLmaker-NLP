# parser --------------------------------------------------------------------------
import argparse
import yaml
from box import Box

parser = argparse.ArgumentParser(description='Configuration to TRBLL model')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to yaml config file. defualt: config.yaml')
args = parser.parse_args()
with open(args.config, encoding="utf8") as f:
    global config_args
    config_args = Box(yaml.load(f, Loader=yaml.FullLoader))

with open('config.yaml') as f:
    global training_args
    training_args = Box(yaml.load(f, Loader=yaml.FullLoader))

with open('private_config.yaml') as f:
    global private_args
    private_args = Box(yaml.load(f, Loader=yaml.FullLoader))