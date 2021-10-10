import copy

import bisect
from typing import Iterable

import re
import sys

from torch.utils.data import Dataset, DataLoader, Subset, IterableDataset
import numpy as np
import torch
from torchvision import transforms
import cv2 as cv
import os

sys.path.append('../../')

from key_modules import transformations

img2npy = lambda x: x.replace('jpg', 'npy').replace('png', 'npy')


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def get_raw_imgs_with_label(self, idx):
        pass

    def get_raw_imgs(self, idx):
        pass


class MultiViewDataset(BaseDataset):
    """
        默认路径下有view1,view2,label,Rt_label 4个文件夹
    """

    def __init__(self, root_dir=r'../blender_dataset', dataset_dir='MultiView_dataset', norm='NORM', label_mode='ALL',
                 transform=None):
        super().__init__()
        assert transform is not None
        self.root_dir = root_dir
        self.norm = norm
        self.labels_dir = os.path.join(root_dir, dataset_dir, r'label')
        self.dataset_path = os.path.join(root_dir, dataset_dir)
        self.dataset_view1_path = os.path.join(self.dataset_path, 'view1')
        self.dataset_view2_path = os.path.join(self.dataset_path, 'view2')
        all_files = list(
            set(os.listdir(self.dataset_view1_path)).intersection(set(os.listdir(self.dataset_view2_path))))
        self.image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
        self.image_files.sort()

        self.transform = transform
        print('Using {} dataset with {} normalization!'.format(dataset_dir, norm))

        self.return_F = False
        self.return_Rt = False
        if label_mode == 'ALL':  # 可以有3种选择， 单独返回Rt或F，以及返回Rt和F
            self.Rt_labels_dir = os.path.join(root_dir, dataset_dir, r'Rt_label')
            print('Using Rt and F together as dataset label')
            self.return_F = True
            self.return_Rt = True
        elif label_mode == 'RT':
            self.Rt_labels_dir = os.path.join(root_dir, dataset_dir, r'Rt_label')
            print('Using Rt as dataset label')
            self.return_Rt = True
        else:
            print('Using F as dataset label')
            self.return_F = True

    def __getitem__(self, index=0):
        # 当前数据集的图片
        image_file = self.image_files[index]
        img_left_path = os.path.join(self.dataset_view1_path, image_file)
        img_right_path = os.path.join(self.dataset_view2_path, image_file)
        img_left = self.transform(self.img_prep(img_left_path))
        img_right = self.transform(self.img_prep(img_right_path))
        if self.return_F:
            F = self.get_F(os.path.join(self.labels_dir, img2npy(image_file)), normalization=self.norm)
            pts1, pts2 = np.load(os.path.join(self.labels_dir, 'points_' + img2npy(image_file)),
                                 allow_pickle=True)
            if self.return_Rt:
                Rt = np.load(os.path.join(self.Rt_labels_dir, 'Rt_' + img2npy(image_file)), allow_pickle=True)
                return torch.cat([img_left, img_right], dim=0), (
                F.astype(np.float32), pts1.T.astype(np.float32), pts2.astype(np.float32), Rt.astype(np.float32))
            return torch.cat([img_left, img_right], dim=0), (
            F.astype(np.float32), pts1.T.astype(np.float32), pts2.astype(np.float32))
        else:
            Rt = np.load(os.path.join(self.Rt_labels_dir, 'Rt_' + img2npy(image_file)), allow_pickle=True)
            return torch.cat([img_left, img_right], dim=0), Rt.astype(np.float32)

    def __len__(self):
        return len(self.image_files)

    def img_prep(self, img_path):
        # img = Image.open(img_path,mode='r')
        img = cv.imread(img_path, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img  # H,W,C

    def get_raw_imgs_with_label(self, index):
        # 当前数据集的图片带标签
        image_file = self.image_files[index]
        img_left_path = os.path.join(self.dataset_view1_path, image_file)
        img_right_path = os.path.join(self.dataset_view2_path, image_file)
        img_left = self.img_prep(img_left_path)
        img_right = self.img_prep(img_right_path)

        F = self.get_F(os.path.join(self.labels_dir, img2npy(image_file)), normalization=self.norm).astype(
            np.float32)
        assert F is not None

        pts1, pts2 = np.load(os.path.join(self.labels_dir, 'points_' + img2npy(image_file)),
                             allow_pickle=True)
        Rt = np.load(os.path.join(self.Rt_labels_dir, 'Rt_' + img2npy(image_file)), allow_pickle=True)
        return img_left, img_right, F, pts1, pts2, Rt

    def get_raw_imgs(self, index):
        # 当前数据集的图片
        image_file = self.image_files[index]
        img_left_path = os.path.join(self.dataset_view1_path, image_file)
        img_right_path = os.path.join(self.dataset_view2_path, image_file)
        img_left = self.img_prep(img_left_path)
        img_right = self.img_prep(img_right_path)

        return img_left, img_right

    @staticmethod
    def get_F(label_path, normalization='NORM'):
        F = np.load(label_path, allow_pickle=True)
        if normalization == 'ETR':
            return F / F[-1, -1]
        elif normalization == 'ABS':
            return F / np.max(np.abs(F))
        else:
            return F / np.linalg.norm(F)
        # assert np.linalg.matrix_rank(F) == 2


def my_subset(dataset: MultiViewDataset, indices):
    '''
    保证subset可以使用原来dataset的一些方法，所以就自己定义一个subset，直接截取image files list的一部分就行了
    :param dataset:
    :param indices:
    :return:
    '''
    subset = copy.deepcopy(dataset) # 深拷贝
    subset.image_files = [subset.image_files[idx] for idx in indices]
    return subset


class MyConcatDataset(BaseDataset):
    '''
    把pytorch的源码拿过来加了个get raw images函数，没做其他改动
    '''
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[BaseDataset]) -> None:
        super(MyConcatDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes

    def get_raw_imgs_with_label(self, idx):
        # 当前数据集的图片
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_raw_imgs_with_label(sample_idx)

    def get_raw_imgs(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_raw_imgs(sample_idx)

def fetch_concat_dataset(params):
    '''
    将params中指定的数据集组合在一起，例如params.dataset='Angle0_30_dataset,Angle30_60_dataset'
    :param params:
    :return:
    '''
    # root_dir = '../../Fundamental Matrix/blender_dataset' # 存放数据集的根目录
    root_dir = '../blender_dataset' # 存放数据集的根目录
    normalization = params.normalization # F的处理方式
    lable_mode = params.label_mode # 数据集返回label的方式

    train_transform, test_transform = transformations.fetch_transform(params)

    dataset = re.split('[ ,]', params.dataset)  # params.dataset可能是含有多个数据集的字符串，如'SmallAngle_dataset,Angle30_60_dataset'

    first_set = dataset[0]
    if os.path.isdir(os.path.join(root_dir, first_set, 'view1')):
        whole_set = MultiViewDataset(root_dir=root_dir, dataset_dir=first_set, norm=normalization,
                                     label_mode=lable_mode, transform=train_transform)
    else:
        raise ValueError('数据集不存在view1和view2文件夹')
    size_set = len(whole_set)
    train_size = int(0.8 * size_set)
    cat_train_dataset = my_subset(whole_set, range(train_size))
    cat_test_dataset = my_subset(whole_set, range(train_size, size_set))

    params.dataset = first_set  # 兼容性代码，保证后面载入模型时不会出现对不上的数据集

    for set in dataset[1:]:
        if os.path.isdir(os.path.join(root_dir, set, 'view1')):
            whole_set = MultiViewDataset(root_dir=root_dir, dataset_dir=set, norm=normalization,
                                         label_mode=lable_mode, transform=train_transform)
        else:
            raise ValueError('数据集不存在view1和view2文件夹')

        size_set = len(whole_set)
        train_size = int(0.8 * size_set)  # 保持每个单独的数据集82分，train和test
        cat_train_dataset = MyConcatDataset([cat_train_dataset, my_subset(whole_set, range(train_size))])  # 合并数据集
        cat_test_dataset = MyConcatDataset(
            [cat_test_dataset, my_subset(whole_set, range(train_size, size_set))])  # 合并数据集

    return cat_train_dataset, cat_test_dataset


def load_data_iter(params=None, gpu_counts=None):
    assert gpu_counts, 'No gpu is available'
    # 解析参数
    assert params is not None, 'Params was not passed in dataloader!'

    cat_train_dataset, cat_test_dataset = fetch_concat_dataset(params=params)

    print('Training dataset size: {}, testing dataset size: {}'.format(len(cat_train_dataset), len(cat_test_dataset)))

    # 在参数设置的时候为了能完全复现实验，我们应保证所有GPU的batchnorm共统计样本数相同
    # 如果是distributed训练，则需要把batch_size/gpu counts
    batch_size = int(params.batch_size)
    num_workers = params.num_workers * gpu_counts  # 因为是单线程，所以需要乘以gpu数量，distributed则不用


    # 因为不是distributed训练模式，只有1个主进程，所以num worker需要乘GPU数量
    train_iter = DataLoader(cat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    test_iter = DataLoader(cat_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           pin_memory=True)

    return train_iter,test_iter


if __name__ == '__main__':
    pass