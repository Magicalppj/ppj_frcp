"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os

from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('-parent','--parent_dir', default='../experiments/loss_Rt_epi',
                    help='Directory containing results of experiments')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    train_metrics_file = os.path.join(parent_dir, 'metrics_train_set.json')
    test_metrics_file = os.path.join(parent_dir, 'metrics_test_set.json')
    if os.path.isfile(train_metrics_file) and os.path.isfile(test_metrics_file):
        with open(train_metrics_file, 'r') as f:
            train_metrics = json.load(f) # {F_score:value,rate:value}
        with open(test_metrics_file, 'r') as f:
            test_metrics = json.load(f)
        merge_metrics = {}
        # 合并训练集和测试集的结果
        for key in train_metrics.keys():
            merge_metrics[key] = str(round(train_metrics[key],3))+'/'+ str(round(test_metrics[key],3))
        metrics[parent_dir] = merge_metrics

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

def synthesize_results_to_md(parent_dir):
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    assert os.path.isdir(parent_dir)
    aggregate_metrics(parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(parent_dir, "results.md")
    with open(save_file, 'w') as f:
        print('Saving results at ',save_file)
        f.write(table)
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    # 搜索parent dir目录下所有存在metrics_train_set.json的实验，绘制成表，输出md文件
    metrics = synthesize_results_to_md(args.parent_dir)
