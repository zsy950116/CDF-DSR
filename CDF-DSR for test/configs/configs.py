import yaml
import argparse


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/configs.yaml', help='Path to the config file.')

# Load experiment setting
opts = parser.parse_args()
params = get_config(opts.config)
