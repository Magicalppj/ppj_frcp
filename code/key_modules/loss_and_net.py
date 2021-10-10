# 训练的神经网络模型
from collections import OrderedDict

import numpy as np
import os
import re
import torch
from torch import nn
import torch.nn.functional as functional
import torchvision
from torchvision import models


def fetch_net_and_loss(params, device, evaluate_mode=False):
    model = loss = None
    model = eval("{}".format(params.model))
    loss = eval("{}".format(params.loss))
    assert model is not None and loss is not None
    criternion = loss().to(device=device)  # loss实例化

    # 网络实例化
    first_dataset_path = re.split('[ ,]', params.dataset)[0]
    camera_matrix_path = os.path.join(r'../blender_dataset', first_dataset_path, 'camera_matrix.npy')  # 相机内参位置
    if evaluate_mode:
        params.normalization = 'ETR'
    net = model(out_features=params.resnet_out_features, output_mode=params.label_mode, norm=params.normalization,
                camera_matrix_path=camera_matrix_path, normalize_R=params.normalize_R).to(device)

    print('Net: {}, loss: {}'.format(params.model, params.loss))

    if params.pretrained:
        # 只在pretrain的条件下freeze
        print('Freeze all layers except linear')
        for name, para in net.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    # BN 换 IN
    elif params.batch_norm_to_instance_norm:
        print('Switch batch norm to instance norm!')
        replace_bn(net)  # 该函数比batch norm to Instance norm快一些

    return net, criternion


def replace_bn(module):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    # for attr_str in dir(module):
    for name, target_attr in module.named_children():
        # target_attr = getattr(module, name)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            new_in = torch.nn.InstanceNorm2d(target_attr.num_features)
            setattr(module, name, new_in)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module)

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()


class loss_epi(nn.Module):
    '''
    用特征点到极线的像素距离为loss
    '''
    def __init__(self):
        super(loss_epi, self).__init__()

    def forward(self, Y_hat, Y):
        """
        loss = abs(x2tFx1)/sqrt(A**2+B**2)
        :param Y_hat: 网络输出，默认Y_hat=(F,Rt_vector)
        :param Y: label=(F,x1,x2t,Rt)
        :return: loss:平均到每个特征点到极线的像素偏差
        """
        _, x1, x2t = Y[:3] # 只用齐次坐标形式的特征点对
        F_pred, _ = Y_hat
        l2_pred = F_pred.matmul(x1)  # 求出极线l2的向量表达式: [A,B,C], shape->(Batch,3,50)

        numerator_pred = torch.abs(
            torch.diagonal(torch.matmul(x2t, l2_pred), dim1=-2, dim2=-1))  # 分子|Ax0+By0+C|   shape->(Batch,N)

        denominator_pred = torch.norm(l2_pred[:, :2], dim=-2)  # 分母(Batch,N)

        pixel_distance_pred = numerator_pred / denominator_pred  # 实际的极线偏差 shape->(Batch,N)

        return pixel_distance_pred.mean()  # 平均到每对特征点

class loss_Rt(nn.Module):
    '''
    单纯的对quaternion+translation一起取L1+L2作为loss
    '''
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
    '''
    quaternion和translation分别赋予不同的权重作为loss
    '''
    def __init__(self, quaternion_weight=10., translation_weight=1.):
        super().__init__()
        self.L2 = nn.MSELoss()
        self.quaternion_weight = quaternion_weight
        self.translation_weight = translation_weight
        print('Using L2 Rt loss with weight R:t = {}:{}'.format(quaternion_weight, translation_weight))

    def forward(self, Y_hat, Y):
        """l1 + l2 loss，带权重"""
        Rt = Y[-1]  # (N,7)
        _, Rt_pred = Y_hat
        quaternion_loss = self.L2(Rt_pred[:, :4], Rt[:, :4]) * self.quaternion_weight
        translation_loss = self.L2(Rt_pred[:, 4:], Rt[:, 4:]) * self.translation_weight

        return quaternion_loss + translation_loss


class loss_Rt_epi(nn.Module):
    '''
    结合对fundamental matrix和对Rt的loss，并加权
    '''
    def __init__(self, Rt_weight=100, epi_weight=1):
        super().__init__()
        self.Rt_loss = loss_Rt()
        self.epi_loss = loss_epi()
        self.Rt_weight = Rt_weight
        self.epi_weight = epi_weight
        print('Using loss with Rt and epi, Rt_weight:{}, epi_weight:{}'.format(Rt_weight, epi_weight))

    def forward(self, Y_hat, Y):
        """l1 + l2 loss，带权重"""
        loss_rt = self.Rt_loss(Y_hat, Y) * self.Rt_weight
        loss_epi = self.epi_loss(Y_hat, Y) * self.epi_weight

        return loss_rt + loss_epi


class loss_Rt_l2_epi(Base):
    '''
    结合对fundamental matrix和对Rt的loss，并加权
    '''
    def __init__(self, Rt_weight=100, epi_weight=1):
        super().__init__()
        self.Rt_loss = loss_Rt_l2_weighted()
        self.epi_loss = loss_epi()
        self.Rt_weight = Rt_weight
        self.epi_weight = epi_weight
        print('Using loss with Rt and epi, Rt_weight:{}, epi_weight:{}'.format(Rt_weight, epi_weight))

    def forward(self, Y_hat, Y):
        """l1 + l2 loss，带权重"""
        loss_rt = self.Rt_loss(Y_hat, Y) * self.Rt_weight
        loss_epi = self.epi_loss(Y_hat, Y) * self.epi_weight
        return loss_rt + loss_epi

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

class RtReconstructionLayer(Base):
    '''
    给定相机内参矩阵的条件下，用quaternion和translation重建出Fundamental matrix
    '''
    def __init__(self, camera_matrix_path=r'../blender_dataset/Original_dataset/camera_matrix.npy'):
        super().__init__()

        print("Using RtReconstruction layer with camera: ", camera_matrix_path)
        camera_intrinsic_inverted = np.linalg.inv(np.load(camera_matrix_path)) # 读取相机内参并求逆，方便后续重建F的运算
        camera_intrinsic_inverted = torch.from_numpy(camera_intrinsic_inverted).float()
        self.camera_intrinsic_inverted = torch.nn.Parameter(camera_intrinsic_inverted, requires_grad=False) # 转为parameter

    def quaternion2rot_matrix(self, quaternion):
        """
        输入(N,4)的quaternion，根据四元数转3x3旋转矩阵的公式实现
        :param quaternion: 
        :return: (N,3,3)旋转矩阵
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
        # print(rotation_matrixs)
        return rotation_matrixs

    def tvector2t_matrix(self, t):
        """
        将t向量转为tx矩阵:(N,3)->(N,3,3)
        :param t:
        :return:(N,3,3)
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
        '''
        输入Rt vector -> shape(N,7)
        :param x:
        :return: F -> shape(N,9)
        '''
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        R = self.quaternion2rot_matrix(x[:, :4])
        tx = self.tvector2t_matrix(x[:, 4:])
        F = self.camera_intrinsic_inverted.T.matmul(tx).matmul(R).matmul(self.camera_intrinsic_inverted)

        # 保持一定的兼容性，Reconstruction输出为(N,9)，而不是(N,3,3)
        return F.view((F.shape[0], -1))


class NormalizationLayer(Base):
    """
    输入(N,9)的tensor，进行不同的normalization,返回(N,3,3)的F矩阵
    """

    def __init__(self, mode='ETR'):
        super(NormalizationLayer, self).__init__()
        self.norm = mode
        print('Creating net using {name} normalization'.format(name=mode))

    def forward(self, X):

        if self.norm == 'ABS':
            maxs = torch.max(torch.abs(X), dim=1, keepdim=True).values
            X = torch.div(X, maxs)

        elif self.norm == 'NORM':
            # 训练时使用
            n = torch.norm(X, dim=1, keepdim=True)
            X = torch.div(X, n)

        else:
            # 评估时使用
            w, h = X.shape
            lastEntries = torch.reshape(X[:, -1], (w, -1))  # 也许需要+1e-8，不过暂时不管
            X = torch.div(X, lastEntries)
        return X.reshape((-1, 3, 3))


class MaxpoolWithIndex(Base):
    '''
    带最大值元素位置下标的池化
    '''
    def __init__(self, kernel_size=3, stride=2):
        super(MaxpoolWithIndex, self).__init__()

        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)

    def forward(self, X):
        w, h = X.shape[-1], X.shape[-2]
        X, indices = self.pooling(X)
        indices = indices.float() / (w * h)
        return torch.cat((indices, X), 1)

class DeepRtResNet(Base):
    """one branch的Resnet34网络"""

    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='ALL',
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
        print('normalize_R={}'.format(normalize_R))
        self.in_channels = in_channels
        self.out_features = out_features  # 决定网络估计的是R，t或者Rt
        print('Net out features: ', out_features)


        self.return_Rt = False
        self.return_F = False
        # 可以有3种选择， 网络输出Rt或F，以及同时输出Rt和F
        # 默认是输出(F,Rt)
        if output_mode == 'ALL':
            print('Net output: Rt and F')
            self.return_F = True
            self.return_Rt = True
        elif output_mode == 'RT':
            print('Net output: Rt')
            self.return_Rt = True
        else:
            print('Net output:F')
            self.return_F = True

        resnet = torchvision.models.resnet34(pretrained=True)
        self.backbone = self._adjust_resnet_io(resnet=resnet)
        self.reconstruct = RtReconstructionLayer(camera_matrix_path=camera_matrix_path)  # 重建层
        self.normalize = NormalizationLayer(mode=norm)

    def forward(self, x, Rt_label=None):
        if len(x.shape) == 3:  # 输入数据的调整以适应当前forward流程
            x = x.view(1, *x.shape)
            if Rt_label is not None:
                Rt_label = Rt_label.view(1, *Rt_label.shape)


        x = self.backbone(x)  # 输出(N,7)

        if self.out_features == 7:

            if self.training:
                if self.normalize_R: # 不再使用normalize_R
                    quaternion_div = torch.norm(x[:, :4], dim=1, keepdim=True)  # 四元数部分归一化
                    quaternion_part = x[:, :4] / quaternion_div
                else:

                    # 默认train通道不对quaternion和translation进行normalize处理
                    quaternion_part = x[:, :4]
                translation_part = x[:, 4:]
            else:
                # eval模式下normalize
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

    def F_from_rt(self, rt_vec):
        F = self.normalize(self.reconstruct(rt_vec))  # 基础矩阵归一化为(N,3,3)
        return F

    def _adjust_resnet_io(self, resnet):
        '''
        调整一下原生resnet的输入输出，因为数据是2张RGB在channel维度上concat在一起的，所以input_channels=6
        :param resnet:
        :return:
        '''
        newConv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                             bias=False)
        if self.in_channels == 6:
            newConv1.weight.data[:, :3] = resnet.conv1.weight.data[:, :]
            newConv1.weight.data[:, 3:6] = resnet.conv1.weight.data[:, :]
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
    '''继承，然后修改backbone为resnet18'''

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
        print('Use PoseNet')


class DeepRtPoseNet_light(DeepRtResNet):
    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='F',
                 camera_matrix_path=r'../blender_dataset/MultiView_dataset/camera_matrix.npy', normalize_R=False
                 ):
        super(DeepRtPoseNet_light, self).__init__(in_channels=in_channels, out_features=out_features, norm=norm,
                                                  output_mode=output_mode,
                                                  camera_matrix_path=camera_matrix_path, normalize_R=normalize_R
                                                  )
        posenet = RelPoseNet_light(out_features=out_features)
        self.backbone = posenet
        print('Use light PoseNet with resnet18')


class DeepRtPoseNet_fc(DeepRtResNet):
    def __init__(self, in_channels=6, out_features=7, norm='ETR', output_mode='F',
                 camera_matrix_path=r'../blender_dataset/MultiView_dataset/camera_matrix.npy', normalize_R=False
                 ):
        super(DeepRtPoseNet_fc, self).__init__(in_channels=in_channels, out_features=out_features, norm=norm,
                                               output_mode=output_mode,
                                               camera_matrix_path=camera_matrix_path, normalize_R=normalize_R
                                               )
        posenet = RelPoseNet_fc(out_features=out_features)
        self.backbone = posenet
        print('Use PoseNet with resnet34 and more fc layers')


class RelPoseNet(Base):
    def __init__(self, out_features=7):
        super().__init__()
        self.backbone, self.concat_fc_layer = self._get_backbone()
        self.net_q_fc = None
        self.net_t_fc = None
        self.out_features = out_features

        # 分开出Rt的全连接层
        if out_features == 7:
            self.net_q_fc = nn.Linear(self.concat_fc_layer.out_features, 4)
            self.net_t_fc = nn.Linear(self.concat_fc_layer.out_features, 3)
        elif out_features == 4:
            self.net_q_fc = nn.Linear(self.concat_fc_layer.out_features, 4)
        else:
            self.net_t_fc = nn.Linear(self.concat_fc_layer.out_features, 3)

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
        '''输入在channel维度上concat的RGB图,(N,6,H,W)'''
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        x1, x2 = x[:, :3], x[:, 3:]  # Resnet 默认输入channel=3，需要拆分
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)

        feat = torch.cat((feat1, feat2), 1)
        if self.net_q_fc and self.net_t_fc:
            t_est = self.net_t_fc(self.dropout(self.concat_fc_layer(feat)))  # (N,3)
            q_est = self.net_q_fc(self.dropout(self.concat_fc_layer(feat)))  # (N,4)
            return torch.cat([q_est, t_est], dim=1)  # 输出(N,7)
        elif self.net_t_fc:
            t_est = self.net_t_fc(self.dropout(self.concat_fc_layer(feat)))
            return t_est
        else:
            q_est = self.net_q_fc(self.dropout(self.concat_fc_layer(feat)))  # (N,4)
            return q_est


class RelPoseNet_light(RelPoseNet):
    def __init__(self, out_features=7):
        super().__init__()
        self.backbone, self.concat_fc_layer = self._get_backbone()

    def _get_backbone(self):
        # backbone, concat_layer = None, None
        backbone = models.resnet18(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer


class RelPoseNet_fc(RelPoseNet):
    '''
    多加几层全连接
    '''
    def __init__(self, out_features=7):
        super().__init__(out_features=out_features)
        cat_out_features = self.concat_fc_layer.out_features
        cat_in_features = self.concat_fc_layer.in_features
        self.concat_fc_layer = nn.Sequential(OrderedDict([

            ("cat_fc1", nn.Linear(in_features=cat_in_features, out_features=4 * cat_in_features)),
            ("cat_fc2", nn.Linear(4 * cat_in_features, 4 * cat_in_features)),
            ("cat_fc3", nn.Linear(4 * cat_in_features, 4 * cat_in_features)),
            ("cat_fc4", nn.Linear(4 * cat_in_features, 1024)),
            ("cat_fc5", nn.Linear(1024, 1024)),
            ("cat_fc6", nn.Linear(1024, cat_out_features))]
        )

        )


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    torch.set_printoptions(precision=12)
    import mydataloader

    testDataset = mydataloader.RtDataset()
    FDataset = mydataloader.FundamentalDataset()
    x, y = testDataset[0]
    torch.autograd.set_detect_anomaly(True)
    net = DeepRtResNet(in_channels=6)
    output = net(x)
    with torch.autograd.detect_anomaly():
        output.sum().backward()
    train_iter, test_iter = mydataloader.load_RtDataset_iter()

    loss = loss_Rt()
    j = 0
    # for x,y in train_iter:
    #     print(loss(net(x),y))
    #     j+=1
    #     if j>5:
    #         break

    # 正向传播验证
    # Rt_rc = RtReconstructionLayer()
    # F = Rt_rc(y)
    # nm = NormalizationLayer()
    # F = nm(F).reshape(3,3)
    # img1, img2, gt, p1, p2 = FDataset.get_raw_imgs_with_label(0)
    # # print(np.abs(np.diagonal(p2.dot(F.numpy().dot(p1.T)))))
    # # print(np.abs(np.diagonal(p2.dot(gt.dot(p1.T)))))
    # print(np.diagonal(p2.dot(np.abs(F.numpy()-gt).dot(p1.T))))
    # from mathutils import Quaternion
    # quat = Quaternion(y[:4].numpy())
    # print(quat.to_matrix())

    # 反向传播
    t = y[4:].view(1, -1)
    t.requires_grad = True
    Rt_rc = RtReconstructionLayer()
    tx = Rt_rc.tvector2t_matrix(t)
    l = tx.sum()
    l.backward()

    quat = y[:4].reshape(1, -1)
    quat.requires_grad = True
    r_matrix = Rt_rc.quaternion2rot_matrix(quat)
    c = r_matrix.sum()
    c.backward()

    y.requires_grad = True
    F = Rt_rc(y)
    out = F.sum()
    out.backward()
