import os

import numpy as np
import torch
import torchvision

import utils
from evaluate import plot_raw_image_pair
from key_modules import mydataloader

if __name__ =="__main__":

    print(os.getcwd())
    experiment_dir = r'../experiments/base' # 指定实验文件夹，目录下应该存在params.json文件

    # 读取训练参数文件
    json_path = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)  # dict file
    print('Evaluating experiment at {}'.format(experiment_dir))


    train_set,test_set = mydataloader.fetch_concat_dataset(params=params)


    # 看输入样本是啥样的
    # 每个单独的数据集样本为6000 or 6400
    start = 400
    index = range(start ,start+10)
    tensor_to_PIL = torchvision.transforms.ToPILImage()
    for i in index:
        img1 =  np.asarray(tensor_to_PIL(train_set[i][0][:3]))
        img2 =  np.asarray(tensor_to_PIL(train_set[i][0][3:]))
        plot_raw_image_pair(img1,img2,title=i)
