"""Perform evaluation"""
import threading
import argparse
import os
from numpy import prod
from subprocess import check_call
import sys

import re

import utils
from synthesize_results import synthesize_results_to_md

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('-parent','--parent_dir', default='../experiments/no_syncBN',
                    help='Directory containing params.json')
parser.add_argument('-d', '--device', default='1',type=str, help="Cuda device to use")



def launch_evaluating_job(experiment_dir,device):
    """Launch training of the mymodel with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """

    # Launch training with this config
    device_count = len(re.findall(r'\d', device))
    device = re.findall(r'\d',device)[0] # 只使用一块gpu
    train_set_cmd = "{python} evaluate.py --experiment_dir={experiment_dir} --device={device} -save -train".format(
        device=device, python=PYTHON, experiment_dir=experiment_dir, device_count=device_count,
    )
    print(train_set_cmd)
    test_set_cmd = "{python} evaluate.py --experiment_dir={experiment_dir} --device={device} -save -test".format(
        device=device, python=PYTHON, experiment_dir=experiment_dir, device_count=device_count,
    )
    print(test_set_cmd)
    check_call(train_set_cmd, shell=True)
    check_call(test_set_cmd, shell=True)


if __name__ == "__main__":
    args = parser.parse_args() # 命令行传入要evaluate的实验父级文件夹，以及评估使用的gpu id

    experiment_dirs = []
    utils.find_all_experiment_dirs(parent_dir_path=args.parent_dir, experiments_list=experiment_dirs)
    threads = []
    for experiment_dir in experiment_dirs:
        t = threading.Thread(target=launch_evaluating_job,kwargs={"experiment_dir":experiment_dir,"device":args.device})
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()
    metrics = synthesize_results_to_md(args.parent_dir)