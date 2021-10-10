import math

import numpy as np
import torch
import warnings
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, LambdaLR, \
    ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18



def fetch_optimizer_and_scheduler(params, net):
    # 优化器设置，正则化
    print('Weight decay: ', params.weight_decay)
    if params.optimizer == 'Adam':
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=params.learning_rate,
                                     weight_decay=params.weight_decay)
    elif params.optimizer == 'SGD':
        optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=params.learning_rate,
                                    weight_decay=params.weight_decay, momentum=0.9)
    else:
        raise ValueError('No optimizer is defined')

    if params.scheduler == 'step_lr_big_step':
        print('Using big step for learning rate decay')
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    elif params.scheduler == 'step_lr_small_step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    elif params.scheduler == 'cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=(params.num_epochs -10) if params.warmup_lr else params.num_epochs)
    elif params.scheduler == 'cosineAnnWarm_T2':
        T = 2
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(np.ceil(params.num_epochs -10)/T if params.warmup_lr else np.ceil(params.num_epochs/T)), T_mult=1, eta_min=1e-6)
    elif params.scheduler == 'cosineAnnWarm_T3':
        T = 3
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(np.ceil(params.num_epochs -10)/T if params.warmup_lr else np.ceil(params.num_epochs/T)), T_mult=1, eta_min=1e-6)
    elif params.scheduler == 'cosineAnnWarm_T4':
        T = 4
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(np.ceil(params.num_epochs -10)/T if params.warmup_lr else np.ceil(params.num_epochs/T)), T_mult=1, eta_min=1e-6)
    elif params.scheduler == "step_cosineAnnWarm_T4":
        T = 4
        scheduler = MyCosineAnnealingWarmRestarts(optimizer, T_0=int(np.ceil(params.num_epochs -10)/T if params.warmup_lr else np.ceil(params.num_epochs/T)), decay_per_step=0.2, eta_min=1e-6)
    elif params.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.96)

    else:
        raise NotImplementedError('Not defined scheduler ')

    print('Learning rate: {}, optimizer: {}, scheduler: {}'.format(params.learning_rate, params.optimizer,
                                                                            params.scheduler))

    if params.warmup_lr:
        print('Start training with learning rate warmed up')
        # 默认预热10个epoch，线性上升学习率
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)
        # 预先走一步，免得初始学习率为0
        optimizer.zero_grad()
        optimizer.step()
        scheduler_warmup.step()
        return optimizer,scheduler_warmup

    return optimizer, scheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class MyCosineAnnealingWarmRestarts(_LRScheduler):
    r"""
    个人修改的stepLR + CosineWarmRestarts

    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, T_0, eta_min=0, decay_per_step=1., last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        self.T_0 = T_0
        self.eta_min = eta_min
        assert decay_per_step<=1,"step decay must be less than 1"
        self.decay_per_step=decay_per_step
        self.step_decay = 1 # 默认step decay为1，随着cosine周期的增加进行变化

        super(MyCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2 * self.step_decay
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0:
                self.T_cur = self.T_cur - self.T_0
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                self.T_cur = epoch % self.T_0
            else:
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        self.step_decay = self.decay_per_step ** int(self.last_epoch/self.T_0)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


if __name__ == "__main__":



    model=resnet18(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=0.001)
    mode='cosineAnn'
    if mode=='cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
    elif mode=='cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)

    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=500, after_scheduler=scheduler)

    # scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.8)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)

    '''
    以T_0=5, T_mult=1为例:
    T_0:学习率第一次回到初始值的epoch位置.
    T_mult:这个控制了学习率回升的速度
        - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
            - 5,10,15,20,25,.......处回到最大值
        - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
            - 5,15,35,75,155,.......处回到最大值
    example:
        T_0=5, T_mult=1
    '''

    plt.figure()
    max_epoch=500
    iters=200
    cur_lr_list = []
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(max_epoch):
        scheduler_warmup.step()
        for batch in range(iters):
            '''
            这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
            那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
            则可以在这里进行.step
            '''
            #scheduler.step(epoch + batch / iters)
            optimizer.step()
        # scheduler.step()
        cur_lr=optimizer.param_groups[-1]['lr']
        cur_lr_list.append(cur_lr)
        print('cur_lr:',cur_lr)

    x_list = list(range(len(cur_lr_list)))
    plt.plot(x_list, cur_lr_list)
    plt.show()