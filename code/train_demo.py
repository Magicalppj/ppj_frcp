import re

import time

import os
import math
import tempfile
import argparse
from torch import nn

import utils
from key_modules import mydataloader
from key_modules.loss_and_net import fetch_net_and_loss
from key_modules.optim import fetch_optimizer_and_scheduler
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate


def parse_args():
    """获取命令行参数"""
    # 网络参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', help='Add this to use the resume model', default=False, type=bool)
    parser.add_argument('-exp', '--experiment_dir', default='../experiments/base', type=str,
                        help='The directory to run a experiment and save results')
    parser.add_argument('--seed',
                        default=230,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('-gpu', '--gpu', default='1,2,3', help='cuda device id')

    parser.add_argument('--world_size', default=3, type=int,
                        help='number of distributed processes')
    args = parser.parse_args()
    return args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 关键，设置可见的cuda
args.world_size = len(re.split('[ ,]', args.gpu))

import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # 读取训练参数
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)  # dict file

    # 设置随机数种子
    utils.setup_seed(args, params)
    print('Traing settings: ', args)
    print('Running experiment at {}'.format(args.experiment_dir))

    device = torch.device(args.device)

    # 加载数据集
    train_iter, test_iter = mydataloader.load_data_iter(params=params,
                                                        gpu_counts=args.world_size)

    # 实例化模型
    net, criternion = fetch_net_and_loss(params=params, device=device)
    # 优化器设置
    optimizer, scheduler = fetch_optimizer_and_scheduler(params, net)

    # 加载预训练参数
    checkpoint_path = os.path.join(args.experiment_dir, 'checkpoint.pth')
    if args.resume and os.path.isfile(checkpoint_path):
        # 在保存模型的基础上继续训练
        print('Continue training on saved mymodel')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 读取checkpoint文件

        # 模型读取
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        try:
            scheduler.load_state_dict(checkpoint["scheduler"])

        except Warning("Using custom loading scheduler"):
            scheduler_dict = scheduler.state_dict()
            state_dict = {k: v for k, v in checkpoint["scheduler"].items() if k in scheduler_dict.keys()}
            scheduler_dict.update(state_dict)
            scheduler.load_state_dict(scheduler_dict)

        # 参数读取
        trained_epoch = checkpoint['trained_epoch']  # 已经训练过的epoch个数
        log_label = checkpoint['log_label']
        min_test_loss = checkpoint['min_test_loss']
    else:
        # 网络在创建模块的时候是默认使用pretrain参数的
        # 如果不使用预训练参数
        if not params.pretrained:

            # xavier初始化网络权重
            for m in net.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)

        # 初始化训练参数
        trained_epoch = 0
        log_label = args.experiment_dir + '-- time: ' + params.time
        min_test_loss = 1e8

    # 转为dataparallel模型
    data_parallel_net = torch.nn.parallel.DataParallel(net, device_ids=list(
        range(args.world_size)))  # 因为设置了cuda visibility，所以默认gpu从0开始
    writer = SummaryWriter('../train_logs')  # code目录下的tensorboard logs dir

    data_parallel_net.train()
    print('Already trained for {} epochs , will train for {} epochs:'.format(trained_epoch, params.num_epochs))
    for epoch in range(trained_epoch, trained_epoch + params.num_epochs):
        last_epoch_time = time.time()

        train_loss = train_one_epoch(data_parallel_net, criternion, optimizer, train_iter, device, epoch,
                                     end_epoch=trained_epoch + params.num_epochs)
        test_loss = evaluate(data_parallel_net, criternion, test_iter, device)
        scheduler.step()

        print('epoch %d, train loss %.6f, test acc %.8f, time %.8f sec, lr %.8f'
              % (epoch + 1, train_loss, test_loss, time.time() - last_epoch_time,
                 optimizer.param_groups[0]['lr']))

        writer.add_scalars(log_label, {'train_loss': train_loss,
                                       'test_loss': test_loss}, epoch + 1)

        if min_test_loss >= test_loss:  # 只记录测试集上表现最好的模型
            min_test_loss = test_loss
            print("Update and save the best mymodel till now!")
            torch.save({'trained_epoch': epoch + 1,
                        'net_state_dict': data_parallel_net.module.state_dict(),  # 保存真实模型的参数，去掉ddp的wrap
                        'optimizer': optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'min_test_loss': min_test_loss,
                        'log_label': log_label}, os.path.join(args.experiment_dir, r'best_checkpoint.pth'))

        if (epoch + 1) % 10 == 0:
            # 固定记录模型
            print("Save this mymodel!")
            torch.save({'trained_epoch': epoch + 1,
                        'net_state_dict': data_parallel_net.module.state_dict(),  # 保存真实模型的参数，去掉ddp的wrap
                        'optimizer': optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'min_test_loss': min_test_loss,
                        'log_label': log_label}, os.path.join(args.experiment_dir, r'checkpoint.pth'))

    writer.close()
