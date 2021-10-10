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

sys.path.append('../')

from create_label import get_sample_sequence, get_label_file
from multi_train_utils.distributed_utils import log_to_console
from mymodel import transformations

img2npy = lambda x: x.replace('jpg', 'npy').replace('png', 'npy')


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def get_raw_imgs_with_label(self, idx):
        pass

    def get_raw_imgs(self, idx):
        pass


class FundamentalDataset(BaseDataset):
    """
        默认路径下有left,right,label三个文件夹，根据data的路径，构建dataset类
        train_dataset和 val_dataset直接从源数据集中28分
    """

    def __init__(self, root_dir=r'../blender_dataset', dataset_dir='Original_dataset', norm='NORM', use_mask=False,
                 use_blender=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.norm = norm
        self.labels_dir = os.path.join(root_dir, dataset_dir, r'label')
        self.dataset_path, self.image_files, sample_sequence = get_sample_sequence(root_dir=root_dir,
                                                                                   dataset_dir=dataset_dir)
        self.sample_sequence = np.unique(sample_sequence, axis=0)  # 去掉重复样本
        assert transform is not None
        self.transform = transform
        self.use_blender = use_blender
        if self.use_blender:
            print('Using blender dataset!')
        self.use_mask = use_mask
        if use_mask:
            print('Using masks!')

    def __getitem__(self, index=0):
        image_pair_sequence = self.sample_sequence[index]
        img1 = self.image_files[image_pair_sequence[0]]
        img2 = self.image_files[image_pair_sequence[1]]
        img_item_path_left = os.path.join(self.dataset_path, img1)
        img_item_path_right = os.path.join(self.dataset_path, img2)

        F = self.get_F(os.path.join(self.labels_dir, get_label_file(img1, img2)), normalization=self.norm).astype(
            np.float32)
        # print(F,type(F))
        assert F is not None

        # img_left = self.transform(self.img_prep(cv.imread(img_item_path_left,1),d,cam_id=2))[:,:size,:size]
        # img_right =  self.transform(self.img_prep(cv.imread(img_item_path_right,1),d,cam_id=3))[:,:size,:size]
        img_left = self.transform(self.img_prep(img_item_path_left))
        img_right = self.transform(self.img_prep(img_item_path_right))
        pts1 = pts2 = None
        # if os.path.exists(img2npy(img_item_path_left)):
        if not self.use_blender:
            pts1, area1 = np.load(img2npy(img_item_path_left), allow_pickle=True)
            pts2, area2 = np.load(img2npy(img_item_path_right), allow_pickle=True)
            # 此处pts1 和 pts2 都是(N,3)的shape
            assert pts1 is not None

            if self.use_mask:  # 生成mask
                mask1 = torch.zeros_like(img_left[:1])
                coor, w, h = area1
                x, y = coor
                mask1[:, y:y + h, x:x + w] = 1

                mask2 = torch.zeros_like(img_right[:1])
                coor, w, h = area2
                x, y = coor
                mask2[:, y:y + h, x:x + w] = 1

                return torch.cat([img_left, mask1, img_right, mask2], dim=0).float(), (
                    torch.from_numpy(F).float(), torch.from_numpy(pts1.T).float(),
                    torch.from_numpy(pts2).float())  # 在C通道上concat
            return torch.cat([img_left, img_right], dim=0).float(), (
                torch.from_numpy(F).float(), torch.from_numpy(pts1.T).float(),
                torch.from_numpy(pts2).float())  # 在C通道上concat

        target_points_index = np.load(os.path.join(self.labels_dir, 'pindex_' + img2npy(self.image_files[index])),
                                      allow_pickle=True)
        pts1, pts2 = np.load(os.path.join(self.labels_dir, 'points_' + get_label_file(img1, img2)), allow_pickle=True)
        return torch.cat([img_left, img_right], dim=0).float(), (
            torch.from_numpy(F).float(), torch.from_numpy(pts1.T).float(),
            torch.from_numpy(pts2).float(), torch.from_numpy(target_points_index).short())  # 在C通道上concat

    def __len__(self):
        return self.sample_sequence.shape[0]

    def img_prep(self, img_path):
        # img = Image.open(img_path,mode='r')
        img = cv.imread(img_path, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img

    def get_raw_imgs_with_label(self, index):
        image_pair_sequence = self.sample_sequence[index]
        img1 = self.image_files[image_pair_sequence[0]]
        img2 = self.image_files[image_pair_sequence[1]]
        img_item_path_left = os.path.join(self.dataset_path, img1)
        img_item_path_right = os.path.join(self.dataset_path, img2)

        label_file = get_label_file(img1, img2)
        F = self.get_F(os.path.join(self.labels_dir, label_file), normalization='ETR')
        assert F is not None
        img_left = self.img_prep(img_item_path_left)
        img_right = self.img_prep(img_item_path_right)
        if not self.use_blender:
            p1 = np.load(img2npy(img_item_path_left), allow_pickle=True)
            p2 = np.load(img2npy(img_item_path_right), allow_pickle=True)
        else:
            p1, p2 = np.load(os.path.join(self.labels_dir, 'points_' + label_file),
                             allow_pickle=True)
        # target_points_index = np.load(os.path.join(self.labels_dir, 'pindex_' + img2npy(self.image_files[index])), allow_pickle=True)
        # return img_left, img_right, F, p1[target_points_index], p2[target_points_index]
        return img_left, img_right, F, p1, p2

    def get_raw_imgs(self, index):
        # 当前数据集的图片
        image_pair_sequence = self.sample_sequence[index]
        img1 = self.image_files[image_pair_sequence[0]]
        img2 = self.image_files[image_pair_sequence[1]]
        img_item_path_left = os.path.join(self.dataset_path, img1)
        img_item_path_right = os.path.join(self.dataset_path, img2)
        img_left = self.img_prep(img_item_path_left)
        img_right = self.img_prep(img_item_path_right)

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


class MultiViewDataset(BaseDataset):
    """
        默认路径下有left,right,label三个文件夹，根据data的路径，构建dataset类
        train_dataset和 val_dataset直接从源数据集中28分
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

        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # rgb
        self.transform = transform
        log_to_console('Using {} dataset with {} normalization!'.format(dataset_dir, norm))

        self.return_F = False
        self.return_Rt = False
        if label_mode == 'ALL':  # 可以有3种选择， 单独返回Rt或F，以及返回Rt和F
            self.Rt_labels_dir = os.path.join(root_dir, dataset_dir, r'Rt_label')
            log_to_console('Using Rt and F together as dataset label')
            self.return_F = True
            self.return_Rt = True
        elif label_mode == 'RT':
            self.Rt_labels_dir = os.path.join(root_dir, dataset_dir, r'Rt_label')
            log_to_console('Using Rt as dataset label')
            self.return_Rt = True
        else:
            log_to_console('Using F as dataset label')
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
        # 当前数据集的图片
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
    保证subset可以使用原来dataset的一些方法，所以就自己定义一个subset
    :param dataset:
    :param indices:
    :return:
    '''
    subset = copy.deepcopy(dataset)
    subset.image_files = [subset.image_files[idx] for idx in indices]
    return subset


class MyConcatDataset(BaseDataset):

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


def load_FundamentalDataset_iter(root_path='../blender_dataset', dataset_dir='Original_dataset', norm='NORM',
                                 batch_size=5, shuffle=True, num_workers=12, use_blender=True,
                                 use_mask=False):
    """加载kitti数据集，8，2分，返回训练用的train_iter和test_iter"""
    print('loading {} with {} normalization'.format(dataset_dir, norm))

    whole_set = FundamentalDataset(root_dir=root_path, dataset_dir=dataset_dir, norm=norm, use_blender=use_blender,
                                   use_mask=use_mask)
    size_set = len(whole_set)
    # size_set = 500

    train_size = int(0.8 * size_set)
    # test_size = size_set - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(whole_set, [train_size, test_size])
    train_dataset = Subset(whole_set, range(train_size))
    test_dataset = my_subset(whole_set, range(train_size, size_set))
    print('Training dataset size: {}, testing dataset size: {}'.format(len(train_dataset), len(test_dataset)))

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                           pin_memory=True)
    return train_iter, test_iter


def load_MultiViewDataset_iter(root_path='../blender_dataset', dataset_dir='MultiView_dataset', norm='NORM',
                               label_mode='RT', batch_size=5, shuffle=True, num_workers=12):
    """加载kitti数据集，8，2分，返回训练用的train_iter和test_iter"""
    log_to_console('loading {} with {} normalization'.format(dataset_dir, norm))

    whole_set = MultiViewDataset(root_dir=root_path, dataset_dir=dataset_dir, norm=norm, label_mode=label_mode)
    size_set = len(whole_set)
    # size_set = 500

    train_size = int(0.8 * size_set)
    # test_size = size_set - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(whole_set, [train_size, test_size])
    train_dataset = my_subset(whole_set, range(train_size))
    test_dataset = my_subset(whole_set, range(train_size, size_set))
    log_to_console('Training dataset size: {}, testing dataset size: {}'.format(len(train_dataset), len(test_dataset)))

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                           pin_memory=True)
    return train_iter, test_iter


class RtDataset(Dataset):
    """回归R，t"""

    def __init__(self, root_dir=r'../blender_dataset', dataset_dir='MultiView_dataset'):
        super(RtDataset, self).__init__()
        self.root_dir = root_dir
        self.labels_dir = os.path.join(root_dir, dataset_dir, r'Rt_label')
        if os.path.isdir(os.path.join(root_dir, dataset_dir, 'view1')):  # 如果存在multiview视角
            self.dataset_path = os.path.join(root_dir, dataset_dir)
            self.dataset_view1_path = os.path.join(self.dataset_path, 'view1')
            self.dataset_view2_path = os.path.join(self.dataset_path, 'view2')
            all_files = list(
                set(os.listdir(self.dataset_view1_path)).intersection(set(os.listdir(self.dataset_view2_path))))
            self.image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
            self.image_files.sort()
            self.sample_sequence = None
        else:
            self.dataset_path, self.image_files, sample_sequence = get_sample_sequence(root_dir=root_dir,
                                                                                       dataset_dir=dataset_dir,
                                                                                       file_name='Rt_sample_sequence.npy')
            self.sample_sequence = np.unique(sample_sequence, axis=0)  # 去掉重复样本

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # rgb

    def __getitem__(self, index=0):
        # print('Loading dataset sample index:',index)
        if self.sample_sequence:
            image_pair_sequence = self.sample_sequence[index]
            img1 = self.image_files[image_pair_sequence[0]]
            img2 = self.image_files[image_pair_sequence[1]]
            img_item_path_left = os.path.join(self.dataset_path, img1)
            img_item_path_right = os.path.join(self.dataset_path, img2)

            # img_left = self.transform(self.img_prep(cv.imread(img_item_path_left,1),d,cam_id=2))[:,:size,:size]
            # img_right =  self.transform(self.img_prep(cv.imread(img_item_path_right,1),d,cam_id=3))[:,:size,:size]
            img_left = self.transform(self.img_prep(img_item_path_left))
            img_right = self.transform(self.img_prep(img_item_path_right))

            Rt = np.load(os.path.join(self.labels_dir, 'Rt_' + get_label_file(img1, img2)), allow_pickle=True)
            return torch.cat([img_left, img_right], dim=0).float(), (torch.from_numpy(Rt).float())
        # 当前数据集的图片
        img_left_path = os.path.join(self.dataset_view1_path, self.image_files[index])
        img_right_path = os.path.join(self.dataset_view2_path, self.image_files[index])
        img_left = self.transform(self.img_prep(img_left_path))
        img_right = self.transform(self.img_prep(img_right_path))
        Rt = np.load(os.path.join(self.labels_dir, 'Rt_' + img2npy(self.image_files[index])), allow_pickle=True)
        return torch.cat([img_left, img_right], dim=0).float(), (torch.from_numpy(Rt).float())

    def img_prep(self, img_path):
        # img = Image.open(img_path,mode='r')
        img = cv.imread(img_path, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img

    def __len__(self):
        if self.sample_sequence:
            return self.sample_sequence.shape[0]
        else:
            return len(self.image_files)

    def get_undistorted_img(self, index):
        if self.sample_sequence:
            image_pair_sequence = self.sample_sequence[index]
            img1 = self.image_files[image_pair_sequence[0]]
            img2 = self.image_files[image_pair_sequence[1]]
            img_item_path_left = os.path.join(self.dataset_path, img1)
            img_item_path_right = os.path.join(self.dataset_path, img2)

            # img_left = self.transform(self.img_prep(cv.imread(img_item_path_left,1),d,cam_id=2))[:,:size,:size]
            # img_right =  self.transform(self.img_prep(cv.imread(img_item_path_right,1),d,cam_id=3))[:,:size,:size]
            img_left = self.img_prep(img_item_path_left)
            img_right = self.img_prep(img_item_path_right)

            Rt = np.load(os.path.join(self.labels_dir, 'Rt_' + get_label_file(img1, img2)), allow_pickle=True)
            return img_left, img_right, Rt
        # 当前数据集的图片
        img_left_path = os.path.join(self.dataset_view1_path, self.image_files[index])
        img_right_path = os.path.join(self.dataset_view2_path, self.image_files[index])
        img_left = self.img_prep(img_left_path)
        img_right = self.img_prep(img_right_path)
        Rt = np.load(os.path.join(self.labels_dir, 'Rt_' + img2npy(self.image_files[index])), allow_pickle=True)
        return img_left, img_right, Rt


def load_RtDataset_iter(root_path='../blender_dataset', dataset_dir='MultiView_dataset', batch_size=5, shuffle=True,
                        num_workers=12):
    """加载Rt数据集，8，2分，返回训练用的train_iter和test_iter"""
    print('loading {} with Rt label '.format(dataset_dir))

    whole_set = RtDataset(root_dir=root_path, dataset_dir=dataset_dir)
    size_set = len(whole_set)
    train_size = int(0.8 * size_set)
    # train_dataset, test_dataset = torch.utils.data.random_split(whole_set, [train_size, test_size]) # 随机2，8分
    train_dataset = my_subset(whole_set, range(train_size))
    test_dataset = my_subset(whole_set, range(train_size, size_set))
    print('Training dataset size: {}, testing dataset size: {}'.format(len(train_dataset), len(test_dataset)))

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                           pin_memory=True)
    return train_iter, test_iter


def fetch_concat_dataset(params):
    root_dir = '../blender_dataset'

    normalization = params.normalization
    lable_mode = params.label_mode
    train_transform, test_transform = transformations.fetch_transform(params)

    dataset = re.split('[ ,]', params.dataset)  # params.dataset可能是含有多个数据集的字符串，如'SmallAngle_dataset,Angle30_60_dataset'

    first_set = dataset[0]
    if os.path.isdir(os.path.join(root_dir, first_set, 'view1')):
        whole_set = MultiViewDataset(root_dir=root_dir, dataset_dir=first_set, norm=normalization,
                                     label_mode=lable_mode, transform=train_transform)
    else:
        whole_set = FundamentalDataset(root_dir=root_dir, dataset_dir=first_set, norm=normalization,
                                       use_blender=True, transform=train_transform)
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
            whole_set = FundamentalDataset(root_dir=root_dir, dataset_dir=set, norm=normalization,
                                           use_blender=True, transform=train_transform)
        size_set = len(whole_set)
        train_size = int(0.8 * size_set)  # 保持每个单独的数据集82分，train和test
        cat_train_dataset = MyConcatDataset([cat_train_dataset, my_subset(whole_set, range(train_size))])  # 合并数据集
        cat_test_dataset = MyConcatDataset(
            [cat_test_dataset, my_subset(whole_set, range(train_size, size_set))])  # 合并数据集

    return cat_train_dataset, cat_test_dataset


def load_distributed_iter(params=None, gpu_counts=None):
    assert gpu_counts, 'No gpu is available'
    # 解析参数
    assert params is not None, 'Params was not passed in dataloader!'

    cat_train_dataset, cat_test_dataset = fetch_concat_dataset(params=params)

    log_to_console('Training dataset size: {}, testing dataset size: {}'.format(len(cat_train_dataset), len(cat_test_dataset)))

    batch_size = int(params.batch_size / gpu_counts)  # 在参数设置的时候为了能完全复现实验，我们应保证所有GPU的batchnorm共统计样本数相同
    num_workers = params.num_workers

    train_sampler = torch.utils.data.distributed.DistributedSampler(cat_train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(cat_test_dataset)
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(cat_train_dataset, num_workers=num_workers,
                                               pin_memory=True, batch_sampler=train_batch_sampler)
    test_loader = torch.utils.data.DataLoader(cat_test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              pin_memory=True, sampler=test_sampler)

    return train_loader, test_loader, train_sampler, test_sampler


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    root_dir = r'../../blender_dataset'
    F_dataset = MultiViewDataset(root_dir, norm='NORM', dataset_dir='SmallAngle_dataset')
    # train_iter, test_iter = load_FundamentalDataset_iter(root_path=root_dir,batch_size=48)
    train_iter, test_iter = load_MultiViewDataset_iter(root_path=root_dir, use_Rt=True,
                                                       dataset_dir='SmallAngle_dataset', shuffle=False, batch_size=60)
    print(len(F_dataset))

    i = 0
    import mynet

    loss = mynet.loss_epi()
    for x, y in train_iter:
        # print(loss(y[0],y))
        print(i)
        i += 1
        if i > 0:
            break
    # img_pair  = x[1].permute((1,2,0))
    # img1, img2 = img_pair[:, :, :3], img_pair[:, :, 3:]
    # img1 = ((img1.cpu().numpy()+1) * 255/2).astype(np.int)
    # img2 = ((img2.cpu().numpy()+1) * 255/2).astype(np.int)
    # _,_, F, p1, p2 = F_dataset.get_raw_imgs_with_label(1)
    # evaluate.plot_polarline(F,img1=img1,img2=img2,p1=p1,p2=p2)
    # x, y = F_dataset[0]
    # img1 = x[:3].permute([1, 2, 0]).numpy()
    # mask1 = x[3].numpy()
