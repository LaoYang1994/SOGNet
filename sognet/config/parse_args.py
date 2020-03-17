import argparse
from .config import config, update_config

def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--eval_only', help='if only eval existing results', action='store_true')
    parser.add_argument('--weight_path', help='manually specify model weights', type=str, default='')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    args = parser.parse_args()
    return args
