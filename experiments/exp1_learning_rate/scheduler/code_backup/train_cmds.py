"""Peform hyperparemeters search"""
import time
from collections import OrderedDict
import argparse
import os
import shutil
from subprocess import check_call
import sys
import re

import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment_id', default=0, type=int,
                    help='Experiment id')
parser.add_argument('-name', '--experiment_name', default='test', type=str,
                    help='Experiment name')
parser.add_argument('-parent', '--parent_dir', default='../experiments/syncBN',
                    help='Directory containing params.json')
parser.add_argument('-cross', '--cross_params', action='store_true',
                    help='Cross params for experiments')
parser.add_argument('-base', '--base_experiment_dir', default=None,
                    help='Directory containing params.json')
parser.add_argument('-r', '--resume', help='Add this to use the newest existing model', default=False, type=bool)

parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('-d', '--device', default='0,1,2', help="Cuda devices to use")

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def parse_var_dict(var_dict: dict):
    var_dict = OrderedDict(var_dict)
    experiment_dir = '+'.join(var_dict.keys())
    return experiment_dir

def init_experiment_dir(var_dict,base_params:utils.Params,args):
    '''
    初始化一个实验的文件夹，备份代码，保存params文件
    :param var_dict:
    :param base_params:
    :param args:
    :return: 实验文件夹的path
    '''
    # cross params 实验父级文件夹
    experiment_parent_dir_name = parse_var_dict(var_dict=var_dict)
    experiment_parent_dir = os.path.join(args.parent_dir, experiment_parent_dir_name)
    if not os.path.isdir(experiment_parent_dir):
        print('Making dir for {var_name} experiments'.format(var_name=experiment_parent_dir_name))
        os.mkdir(experiment_parent_dir)

    # 用拷贝的参数进行改动和实验，保证所有类别超参数的base params json都是一样的
    var_json_path = os.path.join(experiment_parent_dir, 'params.json')
    var_params = base_params._copy() # 深拷贝
    var_params.dict.update({"time": time.asctime(time.localtime())})  # 保存实验时间，方便复盘
    var_params.dict.update({"base_experiment_dir": args.base_experiment_dir})  # 记录base实验文件夹
    var_params.save(var_json_path)  # 储存当前实验文件夹中的base json文件

    utils.backup_code(experiment_parent_dir)  # 开始实验之前备份代码

    return experiment_parent_dir

def check_same_job(param_name, base_params, var_params, exp_path,base_exp_path):
    # 如果该实验和base实验是同一个实验，且该base实验已经有结果了，则直接拷贝文件过来
    # if base_params.dict.get(param_name) == var_params.dict[param_name] and os.path.isfile(
    #         os.path.join(var_params.base_experiment_dir, 'checkpoint.pth')):
    if base_params.dict == var_params.dict and os.path.isfile(
            os.path.join(base_exp_path, 'checkpoint.pth')):
        print('Find same experiments!')
        print('Copying file: from {} to {}'.format(base_exp_path, exp_path))

        # 拷贝文件
        for file in os.listdir(base_exp_path):
            base_experiment_file_path = os.path.join(base_exp_path, file)  # base文件夹中的文件
            if os.path.isfile(base_experiment_file_path):  # 判断是文件
                current_experiment_file_path = os.path.join(exp_path, file)
                shutil.copy(base_experiment_file_path, current_experiment_file_path)
        pass
        var_params.dict.update({"time": time.asctime(time.localtime())})  # 保存实验时间，方便复盘
        var_params.dict.update({"base_experiment_dir": base_exp_path})  # 保存base实验文件夹
        var_params.save(os.path.join(exp_path, 'params.json'))  # 存一个新的params
        # 如果是同一个实验则返回1
        return 1
    # 不是同一个实验，返回0
    var_params.dict.update({"time": time.asctime(time.localtime())})  # 保存实验时间，方便复盘
    var_params.dict.update({"base_experiment_dir": base_exp_path})  # 保存base实验文件夹
    var_params.save(os.path.join(exp_path, 'params.json'))  # 存一个新的params
    return 0

def launch_experiment(parent_dir, var_dict, params, args, base_params):
    var_dict = OrderedDict(var_dict)  # 转为order dict 会进行浅拷贝吧
    if len(var_dict) == 0:
        raise ValueError('Empty hyper params dict!')

    first_param = list(var_dict.keys())[0]  # 取一个hyper param出来
    first_param_value = var_dict.pop(first_param)  # 用拷贝字典

    # hyper params 的取值转为字典
    if not isinstance(first_param_value, list):
        first_param_value = [first_param_value]

    var_params = params._copy()  # 因为是递归，所以可能在调用自身的过程中改动params.dict，使用copy保险

    for var_value in first_param_value:
        exp_dir = first_param + '_' + str(var_value)
        exp_path = os.path.join(parent_dir, exp_dir)
        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)

        # 当前实验的参数更新
        var_params.dict.update({first_param: var_value})  # 更新var params中的参数值
        # var_params.dict.update({"time": time.asctime(time.localtime())})  # 保存实验时间，方便复盘
        # var_params.dict.update({"base_experiment_dir": args.base_experiment_dir})  # 保存base实验文件夹


        if len(var_dict) == 0:
            # 已经到递归末端了，不需要再开子文件夹了
            if not check_same_job(param_name=first_param, base_params=base_params, var_params=var_params,
                                  exp_path=exp_path,base_exp_path=args.base_experiment_dir):
                # 不是同一个实验
                launch_single_training_job(experiment_dir=exp_path, data_dir=None, device=args.device,
                                           params=var_params, resume=args.resume)
        else:
            launch_experiment(parent_dir=exp_path, var_dict=var_dict, params=var_params,
                              base_params=base_params, args=args)

def launch_single_training_job(experiment_dir, data_dir, device, params, resume=False):
    """Launch training of the mymodel with a set of hyperparameters in parent_dir/job_name

    Args:
        experiment_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """

    # Create a new folder in parent_dir with unique_name "job_name"
    if not os.path.isdir(experiment_dir):
        print('Making dir:', experiment_dir)
        os.makedirs(experiment_dir)
    else:
        if os.path.isfile(os.path.join(experiment_dir, 'checkpoint.pth')):
            # 如果在该文件夹中已经进行过实验，则报错，要么手动删除，要么使用预训练参数
            if not args.resume:
                raise RuntimeError(
                    'Already has a experiment in {}! Manually delete the file or use resume'.format(experiment_dir))

    # Write parameters in json file
    json_path = os.path.join(experiment_dir, 'params.json')
    params.save(json_path)

    device_count = len(re.findall(r'\d', device))

    # 用torch distributed launch启动，无法指定gpu
    # cmd = "CUDA_VISIBLE_DEVICES={device} " \
    #       "{python} -m torch.distributed.launch --nproc_per_node={device_count} train_ddp.py --experiment_dir={model_dir} --resume={resume}".format(
    #     device=device, python=PYTHON, model_dir=experiment_dir, device_count=device_count, resume=resume
    # )

    cmd = "CUDA_VISIBLE_DEVICES={device} " \
          "{python} train_multi_gpu_using_spawn.py --experiment_dir={model_dir} --gpu={device} --resume={resume}".format(
        device=device, python=PYTHON, model_dir=experiment_dir, device_count=device_count, resume=resume
    )
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from base_experiment_dir json file


    args.parent_dir = utils.search_experiment(args.experiment_id, args.experiment_name, '../experiments')
    if not args.base_experiment_dir:
        args.base_experiment_dir = args.parent_dir
    base_json_path = os.path.join(args.base_experiment_dir, 'params.json')  # 对照实验目录下的json文件
    assert os.path.isfile(base_json_path), "No json configuration file found at {}".format(base_json_path)
    base_params = utils.Params(base_json_path)

    # Perform hypersearch over each parameter
    var_dict = utils.Params(json_path='var_dict.json').dict  # 从文件中加载本次实验的参数

    if args.cross_params:
        # 默认是不交叉的
        # 使用交叉实验，在var dict json文件中指定多个超参数的时候会进行交叉

        cross_experiment_parent_dir = init_experiment_dir(var_dict, base_params, args)

        launch_experiment(parent_dir=cross_experiment_parent_dir, var_dict=var_dict, params=base_params, base_params=base_params,
                          args=args)

    else:
        for var_name in var_dict.keys():
            # 这个循环是单独处理每种实验超参数，不进行交叉
            # 1. 建一个文件夹开始多组实验
            # 2. 拷贝params.json作为base
            # 3. backup 所有的实验代码
            # 4. 循环开始进行超参数实验

            single_var_dict = {var_name:var_dict[var_name]} # 单超参数的dict
            cross_experiment_parent_dir = init_experiment_dir(single_var_dict, base_params, args)

            launch_experiment(parent_dir=cross_experiment_parent_dir, var_dict=single_var_dict, params=base_params,
                              base_params=base_params,
                              args=args)



            # # 一个参数需要创建一个experiment parent dir
            # var_parent_dir = os.path.join(args.parent_dir, var_name)
            # if not os.path.isdir(var_parent_dir):
            #     print('Making dir for {var_name} experiments'.format(var_name=var_name))
            #     os.mkdir(var_parent_dir)
            #
            # # 用拷贝的参数进行改动和实验，保证所有类别超参数的base params json都是一样的
            # var_json_path = os.path.join(var_parent_dir, 'params.json')
            # shutil.copy(base_json_path, var_json_path)  # 在args.experiment_dir/var_name目录下创建json文件
            #
            # # 当前实验的base
            # var_params = utils.Params(var_json_path)
            # var_params.dict.update({"time": time.asctime(time.localtime())})  # 保存实验时间，方便复盘
            # var_params.dict.update({"base_experiment_dir": args.base_experiment_dir})  # 记录base实验文件夹
            # var_params.save(var_json_path)  # 储存当前实验文件夹中的base json文件
            #
            # # 确保一组超参数设置转为list，即使只有1个
            # if not isinstance(var_dict[var_name], list):
            #     var_dict[var_name] = [var_dict[var_name]]
            #
            # utils.backup_code(var_parent_dir)  # 开始实验之前备份代码
            #
            # # 遍历各取值开始实验
            # for var_value in var_dict[var_name]:
            #     # Modify the relevant parameter in params
            #     # exec ("var_params.dict.update(dict(%s=%s))"%(var_name,var_value)) # 改变var params中的参数值，使用update保证可以添加新键值对
            #
            #     # 当前实验的参数更新
            #     var_params.dict.update({var_name: var_value})  # 更新var params中的参数值
            #     var_params.dict.update({"time": time.asctime(time.localtime())})  # 保存实验时间，方便复盘
            #     var_params.dict.update({"base_experiment_dir": args.base_experiment_dir})  # 保存base实验文件夹
            #
            #     job_name = "{var_name}_{var_value}".format(var_name=var_name,
            #                                                var_value=var_value)  # 例如learning_rate_0.01
            #     current_experiment_dir = os.path.join(var_parent_dir, job_name)  # 当前实验文件夹
            #     if not os.path.isdir(current_experiment_dir):
            #         os.mkdir(current_experiment_dir)  # 创建文件夹
            #
            #     # 如果该实验和base实验是同一个实验，且该base实验已经有结果了，则直接拷贝文件过来
            #     if not check_same_job(param_name=var_name, base_params=base_params,
            #                                  var_params=var_params,
            #                                  exp_path=current_experiment_dir):
            #         # Launch job (name has to be unique)
            #         launch_single_training_job(current_experiment_dir, args.data_dir, args.device, var_params,
            #                                    resume=args.resume)
