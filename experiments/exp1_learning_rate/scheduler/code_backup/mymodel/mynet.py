# 训练的神经网络模型

import numpy as np
import os
import re
import torch
from torch import nn
import torch.nn.functional as functional
import torchvision
from torchvision import models

from multi_train_utils.distributed_utils import log_to_console


def fetch_net_and_loss(params, device, evaluate_mode=False):
    model = loss = None
    model = eval("{}".format(params.model))
    loss = eval("{}".format(params.loss))
    assert model is not None and loss is not None
    criternion = loss().to(device=device)  # loss实例化

    # 网络实例化
    first_dataset_path = re.split('[ ,]',params.dataset)[0]
    camera_matrix_path = os.path.join(r'../blender_dataset',first_dataset_path , 'camera_matrix.npy')  # 相机内参位置
    if evaluate_mode:
        params.normalization = 'ETR'
    net = model(out_features=params.resnet_out_features, output_mode=params.label_mode, norm=params.normalization,
                camera_matrix_path=camera_matrix_path, normalize_R=params.normalize_R).to(device)

    log_to_console('Net: {}, loss: {}'.format(params.model,params.loss))

    # BN 换 IN
    if params.batch_norm_to_instance_norm:
        log_to_console('Switch batch norm to instance norm!')
        # batch_norm_to_instance_norm(net)
        replace_bn(net,'original net') # 新的代码，测测呢

    # 是否冻结权重 注：该参数与换BN的兼容性目前还未知
    # 参考https://www.jianshu.com/p/142e2ab879d3，BN的参数不在梯度回传中，在train()模式下即使requires_grad=False，running mean也会变
    elif params.freeze_layers and params.pretrained:
        # 只在pretrain的条件下freeze
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            log_to_console('Freeze conv layers')
            if "fc" not in name:
                para.requires_grad_(False)

    return net, criternion

def replace_bn(module, name=None):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    # for attr_str in dir(module):
    for name,attr_str in module.named_children():
        # target_attr = getattr(module, attr_str)
        target_attr = getattr(module, name)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            # print('replaced: ', name, attr_str)
            new_bn = torch.nn.InstanceNorm2d(target_attr.num_features)
            # setattr(module, attr_str, new_bn)
            setattr(module, name, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)


def batch_norm_to_instance_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.InstanceNorm2d(num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_instance_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()


class loss_epi(nn.Module):
    def __init__(self):
        super(loss_epi, self).__init__()

    def forward(self, Y_hat, Y, l1_weight=10, l2_weight=1):
        """
        loss = abs(x2tFx1).sum()
        :param Y_hat: 网络输出
        :param Y: 标签
        :param l1_weight:
        :param l2_weight:
        :return: loss
        """
        # _, p1, p2t, index = Y
        # index = torch.where(index)[0]
        # x2t = torch.index_select(p2t, dim=-2, index=index)
        # x1 = torch.index_select(p1, dim=-1, index=index)  # x1 -> shape(3,N)
        _, x1, x2t = Y[:3]
        F_pred, _ = Y_hat
        l2_pred = F_pred.matmul(x1)  # [A,B,C] shape->(Batch,3,50)
        numerator_pred = torch.abs(
            torch.diagonal(torch.matmul(x2t, l2_pred), dim1=-2, dim2=-1))  # 分子|Ax0+By0+C|   shape->(Batch,N)
        denominator_pred = torch.norm(l2_pred[:, :2], dim=-2)  # 分母(Batch,N)
        pixel_distance_pred = numerator_pred / denominator_pred  # 实际的极线偏差 shape->(Batch,N)
        return pixel_distance_pred.mean()  # 平均到每对特征点


class loss_reduce_epi(nn.Module):
    """可以考虑使用物理上的像素到线的距离"""

    def __init__(self):
        super(loss_reduce_epi, self).__init__()

    def forward(self, Y_hat, Y, l1_weight=10, l2_weight=1):
        """训练模型的loss，结合了L1、L2以及对极几何等式的值"""
        F, x1, x2t = Y[:3]
        F_pred, _ = Y_hat
        if len(F.shape) == 2:
            F = F.reshape((1, 3, 3))
        epi1 = torch.diagonal(torch.matmul(torch.matmul(x2t, torch.sub(F_pred, F).abs()), x1), dim1=-2, dim2=-1)

        # epi2 = torch.diagonal(
        #     torch.matmul(torch.matmul(x1.permute([0, 2, 1]), torch.sub(Y_hat.permute([0, 2, 1]), F.permute([0, 2, 1])).abs()),
        #                  # tensor相减绝对值不能少
        #                  # x2t.permute([0, 2, 1])), dim1=-2, dim2=-1)
        #                  x2t.permute([0, 2, 1])), dim1=1, dim2=2) # 加了转置
        # epi_loss = (epi1.sum() / epi1.shape[1] + epi2.sum() / epi2.shape[1]) / 2  # 平均到每个特征点

        # epi_loss = epi1.mean(dim=1).sum()
        # epi_loss = epi1.sum() / epi1.shape[-1]

        # return l1_loss + l2_loss + epi_loss
        return epi1.mean()

        # 下面为用实际像素距离之差为loss
        l2 = F.matmul(x1)  # (batch,3,N)
        numerator = torch.abs(torch.diagonal(torch.matmul(x2t, l2), dim1=-2, dim2=-1))  # 分子(Batch,N)
        denominator = torch.norm(l2[:, :2], dim=1)  # 分母(Batch,N)
        pixel_distance = numerator / denominator  # 实际的极线偏差

        l2_pred = Y_hat.matmul(x1)
        numerator_pred = torch.abs(torch.diagonal(torch.matmul(x2t, l2_pred), dim1=-2, dim2=-1))  # 分子(Batch,N)
        denominator_pred = torch.norm(l2_pred[:, :2], dim=1)  # 分母(Batch,N)
        pixel_distance_pred = numerator_pred / denominator_pred  # 实际的极线偏差

        return torch.sub(pixel_distance_pred, pixel_distance).abs().sum()


class loss_l1_l2(nn.Module):
    def __init__(self, l1_weight=10., l2_weight=1.):
        super().__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.l1_weight = torch.nn.Parameter(torch.tensor(l1_weight), requires_grad=False)
        self.l2_weight = torch.nn.Parameter(torch.tensor(l2_weight), requires_grad=False)

    def forward(self, Y_hat, Y, l1_weight=10, l2_weight=1):
        """l1 + l2 loss，带权重"""
        F = Y[0]
        if len(F.shape) == 2:
            # Y_hat = Y_hat.view(3, 3)
            # F = F.view((1, 3, 3))
            Y_hat = Y_hat.reshape((3, 3))
            # assert not torch.isnan(Y_hat)
            # assert not torch.isnan(F)
            # l1_loss = torch.nn.functional.l1_loss(Y_hat, F) * l1_weight
            # l2_loss = torch.nn.functional.mse_loss(Y_hat, F) * l2_weight
            l1_loss = self.L1(Y_hat, F) * self.l1_weight
            l2_loss = self.L2(Y_hat, F) * self.l2_weight
            return l1_loss + l2_loss
        l1_loss = self.L1(Y_hat, F) * self.l1_weight
        l2_loss = self.L2(Y_hat, F) * self.l2_weight

        return (l1_loss + l2_loss) * Y_hat.shape[0]

class loss_Rt(nn.Module):
    def __init__(self, l1_weight=10., l2_weight=1.):
        super().__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.l1_weight = torch.nn.Parameter(torch.tensor(l1_weight), requires_grad=False)
        self.l2_weight = torch.nn.Parameter(torch.tensor(l2_weight), requires_grad=False)

    def forward(self, Y_hat, Y, l1_weight=10, l2_weight=1):
        """l1 + l2 loss，带权重"""
        Rt = Y[-1]
        _, Rt_pred = Y_hat
        l1_loss = self.L1(Rt_pred, Rt) * self.l1_weight  # 平均到每个样本
        l2_loss = self.L2(Rt_pred, Rt) * self.l2_weight

        return (l1_loss + l2_loss)

class loss_Rt_l2_weighted(nn.Module):
    def __init__(self, quaternion_weight=10., translation_weight=1.):
        super().__init__()
        self.L2 = nn.MSELoss()
        self.quaternion_weight = quaternion_weight
        self.translation_weight = translation_weight
        log_to_console('Using L2 Rt loss with weight R:t = {}:{}'.format(quaternion_weight,translation_weight))

    def forward(self, Y_hat, Y):
        """l1 + l2 loss，带权重"""
        Rt = Y[-1] # (N,7)
        _, Rt_pred = Y_hat
        quaternion_loss = self.L2(Rt_pred[:,:4], Rt[:,:4]) * self.quaternion_weight
        translation_loss = self.L2(Rt_pred[:,4:], Rt[:,4:]) * self.translation_weight

        return quaternion_loss+translation_loss

class loss_Rt_epi(nn.Module):
    def __init__(self, Rt_weight=100, epi_weight=1):
        super().__init__()
        self.Rt_loss = loss_Rt()
        self.epi_loss = loss_epi()
        self.Rt_weight = Rt_weight
        self.epi_weight = epi_weight
        log_to_console('Using loss with Rt and epi, Rt_weight:{}, epi_weight:{}'.format(Rt_weight, epi_weight))

    def forward(self, Y_hat, Y):
        """l1 + l2 loss，带权重"""
        # F, x1nt, x2n, Rt = Y[:4]
        # F_pred, Rt_pred = Y_hat
        # loss_rt = self.Rt_loss(Rt_pred, Rt) * self.Rt_weight
        # loss_epi = self.epi_loss(F_pred, (F, x1nt, x2n)) * self.epi_weight
        loss_rt = self.Rt_loss(Y_hat, Y) * self.Rt_weight
        loss_epi = self.epi_loss(Y_hat, Y) * self.epi_weight

        return loss_rt + loss_epi

class loss_Rt_l2_epi(Base):
    def __init__(self, Rt_weight=100, epi_weight=1):
        super().__init__()
        self.Rt_loss = loss_Rt_l2_weighted()
        self.epi_loss = loss_epi()
        self.Rt_weight = Rt_weight
        self.epi_weight = epi_weight
        log_to_console('Using loss with Rt and epi, Rt_weight:{}, epi_weight:{}'.format(Rt_weight, epi_weight))

    def forward(self, Y_hat, Y):
        """l1 + l2 loss，带权重"""
        # F, x1nt, x2n, Rt = Y[:4]
        # F_pred, Rt_pred = Y_hat
        # loss_rt = self.Rt_loss(Rt_pred, Rt) * self.Rt_weight
        # loss_epi = self.epi_loss(F_pred, (F, x1nt, x2n)) * self.epi_weight
        loss_rt = self.Rt_loss(Y_hat, Y) * self.Rt_weight
        loss_epi = self.epi_loss(Y_hat, Y) * self.epi_weight

        return loss_rt + loss_epi

class Residual(Base):  # 残差基本单元
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return functional.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, stride=1, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=stride))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class GlobalAvgPool2d(Base):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return functional.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(Base):
    """把Global average pool的结果降维:(N,C,1,1)->(N,C)"""

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # X shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class PolarReconstructionLayer(Base):
    """重构F矩阵"""

    def __init__(self):
        super(PolarReconstructionLayer, self).__init__()
        log_to_console('Using polar reconstruction layer')
        # self.alpha = nn.Parameter(torch.rand(1),requires_grad=True)
        # self.beta = nn.Parameter(torch.rand(1),requires_grad=True)

    def forward(self, X):
        """输入(N,8)的tensor,输出(N,9)"""
        # assert X.shape[1] is 8
        # assert X.shape[1] is 6
        # v = X[:,:6]
        # vec = v[:,:3]*self.alpha + v[:,3:]*self.beta

        N = X.shape[0]
        alpha = X[:, -2].view(N, -1)
        beta = X[:, -1].view(N, -1)
        v = X[:, :6]
        vec = v[:, :3] * alpha + v[:, 3:] * beta
        Y = torch.cat([v, vec], dim=1)  # vec 放在前面，秩为二的约束也成立，这样对于α和β就需要取比较小的数了
        return Y


class MatrixReconstructionLayer(Base):
    def __init__(self):
        super(MatrixReconstructionLayer, self).__init__()
        log_to_console("Using MatrixReconstruction layer")
        self.device = 'cpu'

    def get_rotation(self, rx, ry, rz):
        # normalize input?
        R_x = torch.stack([
            *[torch.tensor(1., dtype=torch.float32, device=self.device),
              torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(0., dtype=torch.float32, device=self.device)],
            *[torch.tensor(0., dtype=torch.float32, device=self.device), torch.cos(rx), -torch.sin(rx)],
            *[torch.tensor(0., dtype=torch.float32, device=self.device), torch.sin(rx), torch.cos(rx)]
        ]).view(3, 3)
        # R_x[1:,1:] = [[torch.cos(rx), -torch.sin(rx)],[torch.cos(rx), -torch.sin(rx)]]

        R_y = torch.stack([
            *[torch.cos(ry), torch.tensor(0., dtype=torch.float32, device=self.device), -torch.sin(ry)],
            *[torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(1., dtype=torch.float32, device=self.device),
              torch.tensor(0., dtype=torch.float32, device=self.device)],
            *[torch.sin(ry), torch.tensor(0., dtype=torch.float32, device=self.device), torch.cos(ry)]
        ]).view(3, 3)
        R_z = torch.stack([
            *[torch.cos(rz), -torch.sin(rz), torch.tensor(0., dtype=torch.float32, device=self.device)],
            *[torch.sin(rz), torch.cos(rz), torch.tensor(0., dtype=torch.float32, device=self.device)],
            *[torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(1., dtype=torch.float32, device=self.device)]
        ]).view(3, 3)
        R = torch.matmul(R_x, torch.matmul(R_y, R_z))
        return R

    def get_inv_intrinsic(self, f):
        return torch.stack([
            *[-1 / (f + 1e-8), torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(0., dtype=torch.float32, device=self.device)],
            *[torch.tensor(0., dtype=torch.float32, device=self.device), -1 / (f + 1e-8),
              torch.tensor(0., dtype=torch.float32, device=self.device)],
            *[torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(0., dtype=torch.float32, device=self.device),
              torch.tensor(1., dtype=torch.float32, device=self.device)]
        ]).view(3, 3)

    def get_translate(self, tx, ty, tz):
        return torch.stack([
            *[torch.tensor(0., dtype=torch.float32, device=self.device), -tz, ty],
            *[tz, torch.tensor(0., dtype=torch.float32, device=self.device), -tx],
            *[-ty, tx, torch.tensor(0., dtype=torch.float32, device=self.device)]
        ]).view(3, 3)

    def get_linear_comb(self, f0, f1, f2, f3, f4, f5, cf1, cf2):
        return torch.stack([
            *[f0, f1, f2],
            *[f3, f4, f5],
            *[cf1 * f0 + cf2 * f3, cf1 * f1 + cf2 * f4, cf1 * f2 + cf2 * f5]
        ]).view(3, 3)

    def forward(self, X):
        # X -> (N,8)
        # Note: only need out-dim = 8

        self.device = X.device
        out = torch.empty((X.shape[0], 9), device=self.device)

        for index, x in enumerate(X):
            K1_inv = self.get_inv_intrinsic(x[0])
            K2_inv = self.get_inv_intrinsic(x[1])
            R = self.get_rotation(x[2], x[3], x[4])
            T = self.get_translate(x[5], x[6], x[7])
            F = torch.matmul(K2_inv,
                             torch.matmul(T, torch.matmul(R, K1_inv)))  # 是不是R，t位置反了？
            out[index] = torch.reshape(F, [-1])
        return out

        # to get the last row as linear combination of first two rows
        # new_F = get_linear_comb(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
        # new_F = get_linear_comb(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], x[6], x[7])
        # flat = torch.reshape(new_F, [-1])


class RtReconstructionLayer(Base):
    def __init__(self, camera_matrix_path=r'../blender_dataset/Original_dataset/camera_matrix.npy'):
        super().__init__()
        log_to_console("Using RtReconstruction layer with camera: ", camera_matrix_path)
        camera_intrinsic_inverted = np.linalg.inv(np.load(camera_matrix_path))
        camera_intrinsic_inverted = torch.from_numpy(camera_intrinsic_inverted).float()
        self.camera_intrinsic_inverted = torch.nn.Parameter(camera_intrinsic_inverted, requires_grad=False)

    def quaternion2matrix(self, quaternion):
        """
        输入(N,4)的quaternion
        :param quaternion: 
        :return: 
        """
        rotation_matrixs = torch.empty(quaternion.shape[0], 3, 3, dtype=torch.float32,
                                       device=self.camera_intrinsic_inverted.device)
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]
        rotation_matrixs[:, 0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
        rotation_matrixs[:, 0, 1] = 2 * (x * y - z * w)
        rotation_matrixs[:, 0, 2] = 2 * (x * z + y * w)
        rotation_matrixs[:, 1, 0] = 2 * (x * y + z * w)
        rotation_matrixs[:, 1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
        rotation_matrixs[:, 1, 2] = 2 * (y * z - x * w)
        rotation_matrixs[:, 2, 0] = 2 * (x * z - y * w)
        rotation_matrixs[:, 2, 1] = 2 * (y * z + x * w)
        rotation_matrixs[:, 2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2
        # log_to_console(rotation_matrixs)
        return rotation_matrixs

    def tvector2t_matrix(self, t):
        """
        将t向量转为tx矩阵:(N,3)->(N,3,3)
        :param t:
        :return:
        """
        tx = torch.empty(t.shape[0], 3, 3, dtype=torch.float32, device=self.camera_intrinsic_inverted.device)
        tx[:, 0, 0] = 0
        tx[:, 0, 1] = -t[:, 2]
        tx[:, 0, 2] = t[:, 1]
        tx[:, 1, 0] = t[:, 2]
        tx[:, 1, 1] = 0
        tx[:, 1, 2] = -t[:, 0]
        tx[:, 2, 0] = -t[:, 1]
        tx[:, 2, 1] = t[:, 0]
        tx[:, 2, 2] = 0

        return tx

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        R = self.quaternion2matrix(x[:, :4])
        tx = self.tvector2t_matrix(x[:, 4:])
        F = self.camera_intrinsic_inverted.T.matmul(tx).matmul(R).matmul(self.camera_intrinsic_inverted)
        return F.view((F.shape[0], -1))  # 保持模块一致，Reconstruction输出为(N,9)


class NormalizationLayer(Base):
    """
    输入(N,9)的tensor，进行不同的normalization,返回(N,3,3)的F矩阵
    """

    def __init__(self, mode='ETR'):
        super(NormalizationLayer, self).__init__()
        self.norm = mode
        log_to_console('Creating net using {name} normalization'.format(name=mode))

    def forward(self, X):
        if self.norm == 'ABS':
            maxs = torch.max(torch.abs(X), dim=1, keepdim=True).values
            X = torch.div(X, maxs)
        elif self.norm == 'NORM':
            n = torch.norm(X, dim=1, keepdim=True)
            X = torch.div(X, n)
        else:
            w, h = X.shape
            lastEntries = torch.reshape(X[:, -1], (w, -1))  # 也许需要+1e-8，不过暂时不管
            X = torch.div(X, lastEntries)
        return X.reshape((-1, 3, 3))


class MaxpoolWithIndex(Base):
    def __init__(self, kernel_size=3, stride=2):
        super(MaxpoolWithIndex, self).__init__()

        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)

    def forward(self, X):
        w, h = X.shape[-1], X.shape[-2]
        X, indices = self.pooling(X)
        indices = indices.float() / (w * h)
        return torch.cat((indices, X), 1)


class My_deep_net(Base):
    """双流网络"""

    def __init__(self, in_channels=6, reconstruction='polar', norm='ETR'):
        super(My_deep_net, self).__init__()
        log_to_console('Using My_deep_net')
        self.reconstruction = reconstruction

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            resnet_block(64, 64, 1, first_block=True),
            resnet_block(64, 128, 1),
            resnet_block(128, 128, 1)
        )

        self.resnet = DeepFMatResNet(in_channels=256, reconstruction=reconstruction, norm=norm)

        # self.glob_avg = GlobalAvgPool2d()
        #
        # if reconstruction == 'polar':
        #     self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 6), PolarReconstructionLayer())
        #
        # elif reconstruction == 'mat':
        #     self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 8), MatrixReconstructionLayer())
        #
        # else:
        #     self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 9))
        #
        # self.normalize = NormalizationLayer(mode=norm)

    def forward(self, X):
        """
        正向传播
        :param X: X -> (N,C,2*H,W)
        :return: F->(3,3)
        """
        assert torch.isnan(X).int().sum() == 0

        if len(X.shape) == 3:
            X = X.view(1, *X.shape)
        assert len(X.shape) == 4
        C = X.shape[1] // 2
        conv1_1 = self.conv1(X[:, :C, :, :])
        conv1_2 = self.conv1(X[:, C:, :, :])

        # Conv_out = torch.cat([conv1_1,conv1_2], dim=1)
        res_out = self.resnet(torch.cat([conv1_1, conv1_2], dim=1))
        return res_out
        # avg_out = self.glob_avg(res_out)
        #
        # Y = self.normalize(self.linear_out(avg_out))  # reconstruction and normalization
        # assert torch.isnan(avg_out).int().sum() == 0
        # # 此处输出Y的shape->(N,9)
        #
        # return Y


class My_deep_net2(Base):
    """训练使用的网络，直接出结果F"""

    def __init__(self, in_channels=2, reconstruction='polar', norm='ETR'):
        super(My_deep_net2, self).__init__()
        log_to_console('Using My_deep_net2')

        self.reconstruction = reconstruction

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #
        #     nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.res_net1 = nn.Sequential(
            resnet_block(128, 128, 1, first_block=True),
            resnet_block(128, 256, 1),
            resnet_block(256, 128, 1)
        )
        self.res_net2 = nn.Sequential(
            # resnet_block(512, 512, 1, first_block=True),
            resnet_block(256, 256, 1, first_block=True),
            resnet_block(256, 256, 1),
            resnet_block(256, 128, 1)

        )

        self.glob_avg = GlobalAvgPool2d()
        self.reconstruct = None
        if reconstruction == 'polar':
            self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 512), nn.Linear(512, 8))
            # self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 512), nn.Linear(512, 6))
            self.reconstruct = PolarReconstructionLayer()

        elif reconstruction == 'mat':
            self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 512), nn.Linear(512, 8))
            self.reconstruct = MatrixReconstructionLayer()

        else:
            self.linear_out = nn.Sequential(FlattenLayer(), nn.Linear(128, 512), nn.Linear(512, 9))

        self.normalize = NormalizationLayer(mode=norm)

    def forward(self, X):
        """
        正向传播
        :param X: X -> (N,C,2*H,W)
        :return: F->(3,3)
        """
        assert torch.isnan(X).int().sum() == 0

        if len(X.shape) == 3:
            X = X.view(1, *X.shape)
        assert len(X.shape) == 4
        C = X.shape[1] // 2
        conv1_1 = self.conv1(X[:, :C, :, :])
        conv1_2 = self.conv1(X[:, C:, :, :])
        res1_1 = self.res_net1(conv1_1)
        res1_2 = self.res_net1(conv1_2)
        Conv_out3 = self.res_net2(torch.cat([res1_1, res1_2], dim=1))
        avg_out = self.glob_avg(Conv_out3)
        rec_out = self.linear_out(avg_out)
        if self.reconstruct:
            rec_out = self.reconstruct(rec_out)
        Y = self.normalize(rec_out)  # reconstruction and normalization
        assert torch.isnan(avg_out).int().sum() == 0
        # 此处输出Y的shape->(N,9)

        return Y


class DeepFMatNet(Base):
    def __init__(self, in_channels=4, reconstruction='polar', norm='ETR'):
        super(DeepFMatNet, self).__init__()
        log_to_console('Using DeepFMatNet')
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 2, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((32, 32))

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout(0.37)

        self.fc1 = nn.Linear(256 * 32 * 32, 1024)
        self.reconstruct = None

        if reconstruction == 'polar':
            self.fc = nn.Sequential(
                self.fc1,
                self.dropout3,
                nn.Linear(1024, 512),
                # nn.Linear(512, 6)
                nn.Linear(512, 8)
            )
            self.reconstruct = PolarReconstructionLayer()
        elif reconstruction == 'mat':
            self.fc = nn.Sequential(
                self.fc1,
                self.dropout3,
                nn.Linear(1024, 512),
                nn.Linear(512, 8)
            )
            self.reconstruct = MatrixReconstructionLayer()
        else:
            self.fc = nn.Sequential(
                self.fc1,
                self.dropout3,
                nn.Linear(1024, 512),
                nn.Linear(512, 9)
            )

        self.normLayer = NormalizationLayer(mode=norm)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        w, h = X.shape[-1], X.shape[-2]
        X, indices = nn.functional.max_pool2d(X, 2, return_indices=True)

        indices = indices.float() / (w * h)
        X = torch.cat((indices, X), 1)
        # X = self.conv3(X)
        X = self.avg_pool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.fc(X)
        if self.reconstruct:
            X = self.reconstruct(X)

        X = self.normLayer(X)

        return X


class DeepFMatResNet(Base):
    """单流网络"""

    def __init__(self, in_channels=2, out_features=9, reconstruction='polar', norm='ETR'):
        super(DeepFMatResNet, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=True)

        # pretrained1stConvdWeight = self.backbone.conv1.weight
        newConv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                             bias=False)
        newConv1.weight.data.normal_(0, 0.001)
        if in_channels <= 6:
            log_to_console('Using DeepFMatResNet')
            # newConv1.weight.data = nn.Parameter(pretrained1stConvdWeight[:, :3, :, :])

        self.backbone.conv1 = newConv1

        lastClassifierLayerNm = self.backbone.fc.in_features
        self.reconstruct = None
        if reconstruction == 'polar':
            # newFc = nn.Linear(in_features=lastClassifierLayerNm, out_features=6)
            newFc = nn.Linear(in_features=lastClassifierLayerNm, out_features=8)
            self.reconstruct = PolarReconstructionLayer()
        elif reconstruction == 'mat':
            newFc = nn.Linear(in_features=lastClassifierLayerNm, out_features=8)
            self.reconstruct = MatrixReconstructionLayer()
        else:
            newFc = nn.Linear(in_features=lastClassifierLayerNm, out_features=out_features)

        self.backbone.fc = newFc

        self.normalize = NormalizationLayer(mode=norm)

    def forward(self, X):
        # 输入数据的调整以适应当前数据集
        assert torch.isnan(X).int().sum() == 0

        if len(X.shape) == 3:
            X = X.view(1, *X.shape)
        X = self.backbone(X)
        if self.reconstruct:
            X = self.reconstruct(X)
        X = self.normalize(X)
        assert torch.isnan(X).int().sum() == 0

        return X


class DeepRtResNet(Base):
    """单流网络"""

    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='F',
                 camera_matrix_path=r'../blender_dataset/MultiView_dataset/camera_matrix.npy', normalize_R=False
                 ):
        """
        回归四元数和位移向量
        :param in_channels:
        :param out_features:
        :param norm:
        """
        super(DeepRtResNet, self).__init__()
        self.normalize_R = normalize_R
        log_to_console('normalize_R={}'.format(normalize_R))
        self.in_channels = in_channels
        self.out_features = out_features  # 决定网络估计的是R，t或者Rt
        log_to_console('Net out features: ', out_features)
        self.return_Rt = False
        self.return_F = False
        if output_mode == 'ALL':  # 可以有3种选择， 单独返回Rt或F，以及返回Rt和F
            log_to_console('Net output: Rt and F')
            self.return_F = True
            self.return_Rt = True
        elif output_mode == 'RT':
            log_to_console('Net output: Rt')
            self.return_Rt = True
        else:
            log_to_console('Net output:F')
            self.return_F = True

        resnet = torchvision.models.resnet34(pretrained=True)
        self.backbone = self._adjust_resnet_io(resnet=resnet)
        self.reconstruct = RtReconstructionLayer(camera_matrix_path=camera_matrix_path)  # 重建层
        self.normalize = NormalizationLayer(mode=norm)

    def forward(self, x, Rt_label=None):
        if len(x.shape) == 3:  # 输入数据的调整以适应当前数据集
            x = x.view(1, *x.shape)
            if Rt_label is not None:
                Rt_label = Rt_label.view(1, *Rt_label.shape)
        x = self.backbone(x)  # 输出(N,7)

        if self.out_features == 7:

            if self.training:
                if self.normalize_R:
                    quaternion_div = torch.norm(x[:, :4], dim=1, keepdim=True)  # 四元数部分归一化
                    quaternion_part = x[:, :4] / quaternion_div
                else:
                    quaternion_part = x[:, :4]
                translation_part = x[:, 4:]
            else:
                quaternion_div = torch.norm(x[:, :4], dim=1, keepdim=True)  # 四元数部分归一化
                quaternion_part = x[:, :4] / quaternion_div
                translation_div = torch.norm(x[:, 4:], dim=1, keepdim=True)  # 位移向量部分归一化
                translation_part = x[:, 4:] / translation_div

        elif self.out_features == 4:
            if self.training and not self.normalize_R:
                quaternion_part = x[:, :4]
            else:
                quaternion_div = torch.norm(x[:, :4], dim=1, keepdim=True)  # 四元数部分归一化
                quaternion_part = x[:, :4] / quaternion_div
            translation_part = Rt_label[:, 4:]

        elif self.out_features == 3:
            if self.training:
                translation_part = x
            else:
                translation_div = torch.norm(x, dim=1, keepdim=True)  # 位移向量部分归一化
                translation_part = x / translation_div
            quaternion_part = Rt_label[:, :4]
        else:
            raise ValueError('Resnet output features not match')

        Rt_vector = torch.cat([quaternion_part, translation_part], dim=1)

        if self.return_Rt:
            if self.return_F:
                return self.normalize(self.reconstruct(Rt_vector)), Rt_vector
            return Rt_vector

        return self.normalize(self.reconstruct(Rt_vector))

    def evaluate_Rt_separately(self, x, evaluate_R=True, Rt_label=None):
        if self.out_features != 7:
            raise ValueError
            return

        if len(x.shape) == 3:  # 输入数据的调整以适应当前数据集
            x = x.view(1, *x.shape)
            Rt_label = Rt_label.view(1, *Rt_label.shape)

        self.return_Rt = True
        if self.return_F:
            _, y = self.forward(x)
        else:
            y = self.forward(x)

        if evaluate_R:
            quaternion_part = y[:, :4]
            translation_part = Rt_label[:, 4:]
        else:
            quaternion_part = y[:, 4:]
            translation_part = Rt_label[:, :4]

        Rt_vector = torch.cat([quaternion_part, translation_part], dim=1)

        return self.normalize(self.reconstruct(Rt_vector))

    def F_from_rt(self, rt_vec):
        F = self.normalize(self.reconstruct(rt_vec))  # 基础矩阵归一化为(N,3,3)
        return F

    def _adjust_resnet_io(self, resnet):
        newConv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                             bias=False)
        if self.in_channels == 6:
            newConv1.weight.data[:, :3] =  resnet.conv1.weight.data[:,:]
            newConv1.weight.data[:, 3:6] = resnet.conv1.weight.data[:,:]
        elif self.in_channels == 3:
            newConv1.weight[:, :3] = resnet.conv1
        else:
            raise AttributeError('Resnet got a wrong input channels')
        resnet.conv1 = newConv1
        lastClassifierLayerNm = resnet.fc.in_features
        newFc = nn.Linear(in_features=lastClassifierLayerNm, out_features=self.out_features)
        resnet.fc = newFc
        return resnet


class DeepRtResNet18(DeepRtResNet):
    '''继承，然后修改main_module'''

    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='F',
                 camera_matrix_path=r'../blender_dataset/MultiView_dataset/camera_matrix.npy', normalize_R=False
                 ):
        super(DeepRtResNet18, self).__init__(in_channels=in_channels, out_features=out_features, norm=norm,
                                             output_mode=output_mode,
                                             camera_matrix_path=camera_matrix_path, normalize_R=normalize_R,
                                             )
        resnet = torchvision.models.resnet18(pretrained=True)
        self.backbone = self._adjust_resnet_io(resnet=resnet)


class DeepRtPoseNet(DeepRtResNet):
    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='F',
                 camera_matrix_path=r'../blender_dataset/MultiView_dataset/camera_matrix.npy', normalize_R=False
                 ):
        super(DeepRtPoseNet, self).__init__(in_channels=in_channels, out_features=out_features, norm=norm,
                                            output_mode=output_mode,
                                            camera_matrix_path=camera_matrix_path, normalize_R=normalize_R
                                            )
        posenet = RelPoseNet(out_features=out_features)
        self.backbone = posenet
        log_to_console('Use PoseNet')


class DeepRtSiamNet(DeepRtResNet):
    '''继承，然后修改main_module'''

    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='F',
                 camera_matrix_path=r'../blender_dataset/MultiView_dataset/camera_matrix.npy', size=1,normalize_R=False):
        super(DeepRtSiamNet, self).__init__(in_channels=in_channels, out_features=out_features, norm=norm,
                                            output_mode=output_mode,
                                            camera_matrix_path=camera_matrix_path,normalize_R=normalize_R)
        self.backbone = SiamModule(in_channels=in_channels,out_features=out_features)
        log_to_console('Use SiamNet')


class SiamModule(Base):
    def __init__(self, in_channels=6, out_features=7, size=1):
        super(SiamModule, self).__init__()

        configs = [0, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: in_channels // 2 if x == 0 else x * size, configs))
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            MaxpoolWithIndex(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            MaxpoolWithIndex(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            GlobalAvgPool2d(),  # 全局平均池化
            FlattenLayer()  # 转为(N,C)
        )
        self.regression1 = nn.Sequential(
            nn.Linear(2*configs[5], 512),
            nn.Linear(512, 1024)
        )
        self.regression2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, out_features),
        )
    def forward(self, x):
        '''输入在channel维度上concat的RGB图'''
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        x1, x2 = x[:, :3], x[:, 3:] # Resnet 默认输入channel=3，需要拆分
        feat1 = self.featureExtract(x1)
        feat2 = self.featureExtract(x2)

        feat = torch.cat((feat1, feat2), dim=1)

        Rt = self.regression2(self.regression1(feat)) # (N,out_channels)
        return Rt


class MySiamNet(Base):
    '''two branch 网络'''

    def __init__(self, in_channels=6, reconstruction='polar', norm='ETR', size=1,
                 camera_matrix_path=r'../blender_dataset/SmallAngle_dataset/camera_matrix.npy'):
        super(MySiamNet, self).__init__()
        log_to_console('Using MySiamNet')
        self.camera_matrix_path = camera_matrix_path
        configs = [0, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: in_channels // 2 if x == 0 else x * size, configs))
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.InstanceNorm2d(configs[1]),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            MaxpoolWithIndex(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1] * 2, configs[2], kernel_size=5),
            nn.InstanceNorm2d(configs[2]),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            MaxpoolWithIndex(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2] * 2, configs[3], kernel_size=3),
            nn.InstanceNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.InstanceNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.InstanceNorm2d(configs[5]),
            GlobalAvgPool2d(),  # 全局平均池化
            FlattenLayer()  # 转为(N,C)
        )
        self.regression1 = nn.Sequential(
            nn.Linear(configs[5], 512),
            # nn.Dropout(0.37), #drop层效果太强了，去掉
            nn.Linear(512, 1024)
        )
        self.reconstruct = None
        if reconstruction == 'polar':
            self.regression2 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Linear(1024, 8),
            )
            self.reconstruct = PolarReconstructionLayer()
        else:
            self.regression2 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Linear(1024, 7),
            )
            self.reconstruct = RtReconstructionLayer(camera_matrix_path=camera_matrix_path)

        self.normalize = NormalizationLayer(mode=norm)

    def forward(self, X):
        if len(X.shape) == 3:
            X = X.view(1, *X.shape)
        C = X.shape[1] // 2
        r1 = self.featureExtract(X[:, :C, :, :])
        r1 = self.regression1(r1)

        r2 = self.featureExtract(X[:, C:, :, :])
        r2 = self.regression1(r2)
        regression_out = self.regression2(torch.cat([r1, r2], dim=1))
        if self.reconstruct:
            regression_out = self.reconstruct(regression_out)
        Y = self.normalize(regression_out)  # reconstruction and normalization
        # 此处输出Y的shape->(N,9)

        return Y


class RelPoseNet(Base):
    def __init__(self, out_features=7):
        super().__init__()
        self.backbone, self.concat_layer = self._get_backbone()
        self.net_q_fc = None
        self.net_t_fc = None
        self.out_features = out_features
        if out_features == 7:
            self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
            self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        elif out_features == 4:
            self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        else:
            self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)

        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        # backbone, concat_layer = None, None
        backbone = models.resnet34(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x):
        '''输入在channel维度上concat的RGB图'''
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        x1, x2 = x[:, :3], x[:, 3:] # Resnet 默认输入channel=3，需要拆分
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)

        feat = torch.cat((feat1, feat2), 1)
        if self.net_q_fc and self.net_t_fc:
            t_est = self.net_t_fc(self.dropout(self.concat_layer(feat)))  # (N,3)
            q_est = self.net_q_fc(self.dropout(self.concat_layer(feat)))  # (N,4)
            return torch.cat([q_est, t_est], dim=1)  # 输出(N,7)
        elif self.net_t_fc:
            t_est = self.net_t_fc(self.dropout(self.concat_layer(feat)))
            return t_est
        else:
            q_est = self.net_q_fc(self.dropout(self.concat_layer(feat)))  # (N,4)
            return q_est


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    torch.set_printoptions(precision=12)
    import mydataloader

    testDataset = mydataloader.RtDataset()
    FDataset = mydataloader.FundamentalDataset()
    x, y = testDataset[0]
    torch.autograd.set_detect_anomaly(True)
    net = DeepRtResNet(in_channels=6, use_reconstruction=True)
    output = net(x)
    with torch.autograd.detect_anomaly():
        output.sum().backward()
    train_iter, test_iter = mydataloader.load_RtDataset_iter()

    loss = loss_Rt()
    j = 0
    # for x,y in train_iter:
    #     log_to_console(loss(net(x),y))
    #     j+=1
    #     if j>5:
    #         break

    # 正向传播验证
    # Rt_rc = RtReconstructionLayer()
    # F = Rt_rc(y)
    # nm = NormalizationLayer()
    # F = nm(F).reshape(3,3)
    # img1, img2, gt, p1, p2 = FDataset.get_raw_imgs_with_label(0)
    # # log_to_console(np.abs(np.diagonal(p2.dot(F.numpy().dot(p1.T)))))
    # # log_to_console(np.abs(np.diagonal(p2.dot(gt.dot(p1.T)))))
    # log_to_console(np.diagonal(p2.dot(np.abs(F.numpy()-gt).dot(p1.T))))
    # from mathutils import Quaternion
    # quat = Quaternion(y[:4].numpy())
    # log_to_console(quat.to_matrix())

    # 反向传播
    t = y[4:].view(1, -1)
    t.requires_grad = True
    Rt_rc = RtReconstructionLayer()
    tx = Rt_rc.tvector2t_matrix(t)
    l = tx.sum()
    l.backward()

    quat = y[:4].reshape(1, -1)
    quat.requires_grad = True
    r_matrix = Rt_rc.quaternion2matrix(quat)
    c = r_matrix.sum()
    c.backward()

    y.requires_grad = True
    F = Rt_rc(y)
    out = F.sum()
    out.backward()
