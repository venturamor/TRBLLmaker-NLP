# parser --------------------------------------------------------------------------
import argparse
import yaml

parser = argparse.ArgumentParser(description='Configuration to TRBLL model')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to yaml config file. defualt: config.yaml')
args = parser.parse_args()
with open(args.config, encoding="utf8") as f:
    global config_args
    config_args = yaml.load(f, Loader=yaml.FullLoader)