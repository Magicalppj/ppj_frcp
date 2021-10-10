import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from torch import nn
import argparse

from mymodel import mydataloader, mynet

net = mynet.DeepRtResNet


def parse_args():
    """获取命令行参数"""
    # 网络参数

    parser = argparse.ArgumentParser(prog='The main training program', usage="Run it to train",
                                     description='Need parameters:resume, lr , epoch , norm ')
    # parser.add_argument('--s',type=str,default='test_default',help='try it for some information')
    parser.add_argument('-r', '--resume', help='Add this to use the saved mymodel', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-nm', '--normalization', type=str, default='NORM', choices=['NORM', 'ABS', 'ETR'],
                        help='Training principle for normalizing the F matrix')
    parser.add_argument('-d', '--device', type=str,
                        help='The GPU devices used for training: default input like "0 1 2 3" ', default='2,3')
    parser.add_argument('-rc', '--reconstruction', type=str, default='polar', choices=['polar', 'mat', 'none','rt'],
                        help='The reconstruction layer used for the net')
    parser.add_argument('-ep', '--epoch', type=int, default=200, help='Number of epochs to be trained')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='The batch size used for training')
    parser.add_argument('-s', '--start', help='The flag for starting the training process', action='store_true')
    parser.add_argument('-dt', '--dataset', type=str, default='MultiView_dataset',
                        help='The datasets for training')

    parser.add_argument('-nw', '--num_workers', type=int, default=12, help='Num of Dataloader workers')
    parser.add_argument('-ub', '--use_best_model', help='Choose the saved mymodel to be trained', action='store_true')
    parser.add_argument('-mask', '--use_mask', help='Use masks for chessboard images', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-epi', '--use_epi_loss', help='Use epipolar loss to train', action='store_true')
    group.add_argument('-Rt', '--use_Rt_loss', help='Use Rt loss to train', action='store_true')
    group.add_argument('-reduce_epi', '--use_reduce_epi_loss', help='Use reduced epipolar loss to train',
                       action='store_true')

    parser.add_argument('--l1_weight', default=10., type=float,
                        help="Weight for L1-loss")
    parser.add_argument('--l2_weight', default=1., type=float,
                        help='Weight for L2-loss')

    # distributed training
    # parser.add_argument("--rank", default=0, type=int,
    #                     help="node rank for distributed training")
    # parser.add_argument("--dist_url", default=None, type=str,
    #                     help="url used to set up distributed training")
    # parser.add_argument("--dist_backend", default="nccl", type=str)

    args = parser.parse_args()
    return args


def evaluate_loss(data_iter, net, loss, device, l1_weight, l2_weight):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的 device
        device = list(net.parameters())[0].device
    l_sum, batch_count, n = 0.0, 0, 0
    with torch.no_grad():
        net.eval()  # 评估模式, 这会关闭dropout
        for X, y in data_iter:
            X = X.to(device)
            if args.use_Rt_loss:
                y = y.to(device)  # Rt dataset
            else:
                y = [i.to(device).float() for i in y]
            y_hat = net(X)
            l = loss(y_hat, y, l1_weight, l2_weight)
            l_sum += l.cpu().item()
            n += X.shape[0]  # 记录样本的个数
        net.train()  # 改回训练模式
    return l_sum / n  # 返回平均的误差


def parallel_train(train_iter, test_iter, net, loss_module, optimizer, device, training_parameters,
                   dataset_dir='Original_dataset'):
    """
    训练的主程序函数
    :param train_iter:
    :param test_iter:
    :param net:
    :param loss_module:
    :param optimizer:
    :param device:
    :param training_parameters:
    :return:
    """
    lr = training_parameters['lr']
    num_epochs = training_parameters['num_epochs']
    resume = training_parameters['resume']
    loss = loss_module().to(device)

    if resume:
        # 在保存模型的基础上继续训练
        print('Continue training on saved mymodel')

        if training_parameters['use_best_model']:
            print('Reloading the best mymodel:\n')
            # checkpoint = torch.load(r'mymodel/best_checkpoint.pth', map_location=device)
            checkpoint = torch.load(r'mymodel/best_checkpoint.pth', map_location='cpu')

        else:
            print('Reloading the newest mymodel\n')
            # checkpoint = torch.load(r'mymodel/checkpoint.pth', map_location=device)
            checkpoint = torch.load(r'mymodel/checkpoint.pth', map_location='cpu')
        net.to('cpu')
        net.load_state_dict(checkpoint['net_state_dict'])
        net.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        trained_epoch = checkpoint['trained_epoch']
        log_label = checkpoint['log_label']
        min_test_loss = checkpoint['min_test_loss']


    else:
        # 重新训练，使用xavier初始化参数
        print('Retraining:')
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        trained_epoch = 0
        log_label = dataset_dir + ' ' + net.module._get_name() + ' ' + loss._get_name() + '  start time: ' + time.asctime(
            time.localtime())
        min_test_loss = 1e8

    print('Set the learning rate:{}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    net = net.to(device)

    print("training on ", device)
    print('Already trained for {} epochs , will train for {} epochs:'.format(trained_epoch, num_epochs))

    writer = SummaryWriter('../../train_logs')
    l1_weight = training_parameters['l1_weight']
    l2_weight = training_parameters['l2_weight']

    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    net.train()
    for epoch in range(trained_epoch, trained_epoch + num_epochs):
        batch_count = 0
        train_l_sum, n, start = 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device, dtype=torch.float)
            if args.use_Rt_loss:
                y = y.to(device,dtype=torch.float)  # Rt  dataset
            else:
                y = [i.to(device,dtype=torch.float) for i in y]

            y_hat = net(X)
            print(y_hat.device,y.device)

            l = loss(y_hat, y, l1_weight, l2_weight)
            # l = loss(y_hat, y[0])
            # print(l)
            optimizer.zero_grad()
            l.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2)
            # nn.utils.clip_grad_value_(net.parameters(), 20)
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += X.shape[0]
            batch_count += 1
        test_loss = evaluate_loss(test_iter, net, loss, device, l1_weight, l2_weight)
        scheduler.step()

        print('epoch %d, loss %.6f, train acc %.8f, test acc %.8f, time %.8f sec, lr %.8f'
              % (epoch + 1, train_l_sum / batch_count, train_l_sum / n, test_loss, time.time() - start,
                 optimizer.param_groups[0]['lr']))

        writer.add_scalars(log_label, {'loss': train_l_sum / batch_count,
                                       'train_loss': train_l_sum / n,
                                       'test_loss': test_loss}, epoch + 1)

        if min_test_loss >= test_loss:  # 只记录测试集上表现最好的模型
            min_test_loss = test_loss
            print("Update and save the best mymodel till now!")
            torch.save({'trained_epoch': epoch + 1,
                        'net_state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'min_test_loss': min_test_loss,
                        'log_label': log_label}, r'mymodel/best_checkpoint.pth')

        if (epoch + 1) % 20 == 0:
            # 固定记录模型
            print("Save this mymodel!")
            torch.save({'trained_epoch': epoch + 1,
                        'net_state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'min_test_loss': min_test_loss,
                        'log_label': log_label}, r'mymodel/checkpoint.pth')

    writer.close()


if __name__ == '__main__':
    # gpu_id = []
    # os.environ["CUDA_VISBLE_DEVICES"] = str(gpu_id)

    args = parse_args()

    # 对命令行参数的一些预处理
    import re

    args.device = list(map(int, re.findall(r'\d', args.device)))

    print('Hyperparameters :', args)

    # dist.init_process_group(backend=args.dist_backend)
    # torch.cuda.set_device(args.local_rank)

    device = torch.device('cuda:{}'.format(args.device[0]) if torch.cuda.is_available() else 'cpu')
    normalization = args.normalization
    reconstruction = args.reconstruction

    # 训练参数
    lr, num_epochs = args.learning_rate, args.epoch
    resume = args.resume

    training_parameters = {'lr': lr,
                           'num_epochs': num_epochs,
                           'resume': resume,
                           'use_best_model': args.use_best_model,
                           'l1_weight': args.l1_weight,
                           'l2_weight': args.l2_weight
                           }

    if args.start:
        root_path = r'../../blender_dataset'
        net_instance = net()
        torch.cuda.empty_cache()
        print('=====================Starting training=========================')
        if args.use_Rt_loss:
            # train_iter, test_iter = mydataloader.load_RtDataset_iter(root_path=r'../blender_dataset',
            #                                                          dataset_dir=args.dataset,
            #                                                          batch_size=args.batch_size,
            #                                                          num_workers=args.num_workers)
            train_iter, test_iter = mydataloader.load_MultiViewDataset_iter(root_path=r'../../blender_dataset',
                                                                            dataset_dir=args.dataset,
                                                                            norm=normalization,
                                                                            batch_size=args.batch_size,
                                                                            num_workers=args.num_workers, label_mode='RT'
                                                                            )
            net = net(in_channels=6,out_features=7, norm=args.normalization, output_mode='RT',
                      camera_matrix_path=os.path.join(root_path, args.dataset, 'camera_matrix.npy'))

        elif os.path.isdir(os.path.join(root_path, args.dataset, 'view1')):  # 如果存在multiview视角
            train_iter, test_iter = mydataloader.load_MultiViewDataset_iter(root_path=r'../../blender_dataset',
                                                                            dataset_dir=args.dataset,
                                                                            norm=normalization,
                                                                            batch_size=args.batch_size,
                                                                            num_workers=args.num_workers, label_mode='RT'
                                                                            )
            # print(dir(net))
            if hasattr(net_instance, 'use_reconstruction'):
                net = net(in_channels=6, out_features=7,output_mode='RT', norm=args.normalization,
                          camera_matrix_path=os.path.join(root_path, args.dataset, 'camera_matrix.npy'))
            elif hasattr(net_instance, 'camera_matrix_path'):
                net = net(in_channels=6, reconstruction=args.reconstruction, norm=args.normalization,camera_matrix_path=os.path.join(root_path, args.dataset, 'camera_matrix.npy'))
            else:
                net = net(in_channels=6, reconstruction=args.reconstruction, norm=args.normalization)
        else:
            train_iter, test_iter = mydataloader.load_FundamentalDataset_iter(root_path=r'../../blender_dataset',
                                                                              dataset_dir=args.dataset,
                                                                              norm=normalization,
                                                                              batch_size=args.batch_size,
                                                                              num_workers=args.num_workers,
                                                                              use_blender=True,
                                                                              use_mask=args.use_mask)
            if hasattr(net_instance, 'use_reconstruction'):
                net = net(in_channels=6, use_reconstruction=True, norm=args.normalization,
                          camera_matrix_path=os.path.join(root_path, args.dataset, 'camera_matrix.npy'))
            elif hasattr(net_instance, 'camera_matrix_path'):
                net = net(in_channels=6, reconstruction=args.reconstruction, norm=args.normalization,camera_matrix_path=os.path.join(root_path, args.dataset, 'camera_matrix.npy'))
            else:
                net = net(in_channels=6, reconstruction=args.reconstruction, norm=args.normalization)

        Parallel_net = torch.nn.DataParallel(net, device_ids=args.device)
        optimizer = torch.optim.Adam(Parallel_net.parameters(), lr=lr)
        if args.use_epi_loss:
            parallel_train(train_iter, test_iter, Parallel_net, mynet.loss_epi, optimizer, device, training_parameters,
                           args.dataset)
        elif args.use_reduce_epi_loss:
            parallel_train(train_iter, test_iter, Parallel_net, mynet.loss_reduce_epi, optimizer, device,
                           training_parameters, args.dataset)
        elif args.use_Rt_loss:
            parallel_train(train_iter, test_iter, Parallel_net, mynet.loss_Rt, optimizer, device, training_parameters,
                           args.dataset)
        else:
            parallel_train(train_iter, test_iter, Parallel_net, mynet.loss_l1_l2, optimizer, device, training_parameters,
                           args.dataset)
