import math
import random
import tempfile
import time
import argparse
import os
import warnings
from torch.backends import cudnn

import re
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from mymodel import mydataloader, mynet
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup, log_to_console
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
import utils
from mymodel.mynet import fetch_net_and_loss
from mymodel.optim import fetch_optimizer_and_scheduler


def parse_args():
    """获取命令行参数"""
    # 网络参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('-exp', '--experiment_dir', default='../experiments/base', type=str,
                        help='The directory to run a experiment and save results')
    parser.add_argument('--seed',
                        default=230,
                        type=int,
                        help='seed for initializing training. ')
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-r', '--resume', help='Add this to use the resume model', default=False, type=bool)
    args = parser.parse_args()
    return args


args = parse_args()

if __name__ == '__main__':
    # 初始化分布式训练，设置backend,group,rank等
    init_distributed_mode(args=args)

    # 读取训练参数
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)  # dict file

    # 设置随机数种子
    utils.setup_seed(args, params)
    log_to_console('Traing settings: ', args)
    log_to_console('Running experiment at {}'.format(args.experiment_dir))

    rank = args.rank
    device = torch.device(args.device)
    # params.learning_rate *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    # 加载数据集
    train_iter, test_iter, train_sampler, test_sampler = mydataloader.load_distributed_iter(params=params,
                                                                                            gpu_counts=args.world_size)

    # 实例化模型
    net, criternion = fetch_net_and_loss(params=params, device=device)
    # 优化器设置
    optimizer,scheduler = fetch_optimizer_and_scheduler(params, net)

    # 加载预训练参数
    checkpoint_path = os.path.join(args.experiment_dir, 'checkpoint.pth')
    if args.resume and os.path.isfile(checkpoint_path):
        # 在保存模型的基础上继续训练
        log_to_console('Continue training on saved mymodel')
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
        trained_epoch = checkpoint['trained_epoch']
        log_label = checkpoint['log_label']
        min_test_loss = checkpoint['min_test_loss']
    else:
        # 网络在创建模块的时候是默认使用pretrain参数的
        # 如果不使用预训练参数
        if not params.pretrained:

            weights_path = os.path.join(tempfile.gettempdir(), "initial_weights.pth")
            # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致

            if rank == 0:
                for m in net.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
                torch.save(net.state_dict(), weights_path)
                print('Creating temp parameters for net')

            dist.barrier()
            # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
            net.load_state_dict(torch.load(weights_path, map_location='cpu'))

            # 删除临时权重文件
            if rank == 0:
                if os.path.exists(weights_path) is True:
                    os.remove(weights_path)

        # 初始化训练参数
        trained_epoch = 0
        log_label = args.experiment_dir + '-- time: ' + params.time
        min_test_loss = 1e8

    # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    if args.syncBN and not params.batch_norm_to_instance_norm:
        # 使用SyncBatchNorm后训练会更耗时
        log_to_console('SyncBatchNorm while training net')
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)  # 需要在转为DDP模型之前转为syncBN

    # 转为DDP模型
    print('Check device:{}, and gpu:{}'.format(device,args.gpu))
    dist_net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    writer = SummaryWriter('../train_logs')  # code目录下的tensorboard logs dir

    dist_net.train()
    train_loss = 0.0
    log_to_console('Already trained for {} epochs , will train for {} epochs:'.format(trained_epoch, params.num_epochs))
    for epoch in range(trained_epoch, trained_epoch + params.num_epochs):
        last_epoch_time = time.time()
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(dist_net, criternion, optimizer, train_iter, device, epoch,
                                     end_epoch=trained_epoch + params.num_epochs)
        test_loss = evaluate(dist_net, criternion, test_iter, device)
        scheduler.step()

        if rank == 0:
            print('epoch %d, train loss %.6f, test acc %.8f, time %.8f sec, lr %.8f'
                  % (epoch + 1, train_loss, test_loss, time.time() - last_epoch_time,
                     optimizer.param_groups[0]['lr']))

            writer.add_scalars(log_label, {'train_loss': train_loss,
                                           'test_loss': test_loss}, epoch + 1)

            if min_test_loss >= test_loss:  # 只记录测试集上表现最好的模型
                min_test_loss = test_loss
                print("Update and save the best mymodel till now!")
                torch.save({'trained_epoch': epoch + 1,
                            'net_state_dict': dist_net.module.state_dict(),  # 保存真实模型的参数，去掉ddp的wrap
                            'optimizer': optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            'min_test_loss': min_test_loss,
                            'log_label': log_label}, os.path.join(args.experiment_dir, r'best_checkpoint.pth'))

            if (epoch + 1) % 10 == 0:
                # 固定记录模型
                print("Save this mymodel!")
                torch.save({'trained_epoch': epoch + 1,
                            'net_state_dict': dist_net.module.state_dict(),  # 保存真实模型的参数，去掉ddp的wrap
                            'optimizer': optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            'min_test_loss': min_test_loss,
                            'log_label': log_label}, os.path.join(args.experiment_dir, r'checkpoint.pth'))

    writer.close()

    cleanup()
