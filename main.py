import argparse
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--verbose",type=str,default="info",help="Verbose level: info | debug | warning | critical",)
    parser.add_argument("--output",type=str,default="images",help="The folder name of samples",)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta",type=float,default=0.0,help="eta used to control the variances of sigma",)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--mode",type=int,default=0,help="0:Unmodified, 1:Modified")
    parser.add_argument("--collect", action="store_true", help="Collect h?")
    parser.add_argument("--phase_3", action="store_true", help="Run Phase 3?")
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f: config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict): new_value = dict2namespace(value)
        else: new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    runner = Diffusion(args, config)
    runner.sample(args.mode,args.phase_3)
    return 0

if __name__ == "__main__": sys.exit(main())