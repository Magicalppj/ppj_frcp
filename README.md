# Relative Camera Pose Estimation
## 前言 
上传到github上的项目是简化版本，主要的删减改动包括：
1. 去掉了个人使用的大部分框架内容，只保留了一个train demo用作训练网络示例
2. 去掉了分布式训练的多线程代码，用基础的DataParallel进行演示
3. 去掉了不常用的model，loss，dataset等模块

在Project/experiments/exp1_learning_rate/scheduler目录下保存了我之前的实验结果以及params.json文件

如果要进行复现验证的话可以使用相应的params.json文件

|                | F_score     | net_better_performance_rate   |
|:-------------|:------------|:----------------|
| step_lr_small_step    | 0.02/3.844  | 0.92/0.224                    |
| cosineAnn             | 0.0/2.174   | 1.0/0.204                     |
| cosineAnnWarm_T4      | 0.024/1.049 | 0.91/0.213                    |
| step_cosineAnnWarm_T4 | 0.007/1.169 | 0.963/0.215                   |

## 数据集
用于验证实验结果，目前只在Google drive上准备了两个数据集
1. Angle0_30_dataset(代码中可能命名为SmallAngle_dataset)
2. Angle30_60_dataset

**数据集需要放在Project/blender_dataset目录下**

**数据集的基本组织结构**
````
blender_dataset
    ├─Angle0_30_dataset
    └─Angle30_60_dataset
            ├─label
            ├─Rt_label
            ├─samples
            ├─view1
            ├─view2
            └─camera_matrix.npy
````
* 其中view1和view2存放png图片及每张图片对应的世界坐标系到图片的变换矩阵json文件。图片命名和json文件是对应的。

* camera_matrix.npy为相机内参矩阵
* samples是数据集的样本的示例图片
* label以及Rt_label则分别存放F,feature points和Rt的标签
* 数据集目录下的pkl文件是储存blender的material和object信息的，不需要使用


**数据集的使用方式**

读取每个图片相对于世界坐标的外参矩阵，通过矩阵求法求出两个视角相机的相对位姿变换，生成F label及Rt label
```python
        with open(os.path.join(view1_dir_path,'000001.json'), 'r') as file:
            si, projection_matrix, p1 = json.load(file)  # si 是从世界坐标转到相机坐标的外参矩阵
            si = np.asarray(si, dtype=np.float32) # 外参矩阵->shape(4,4)
            p1 = np.asarray(p1, dtype=np.float32) # 特征点

        with open(os.path.join(view2_dir_path,'000001.json'), 'r') as file:
            sj, _, p2 = json.load(file)
            sj = np.asarray(sj, dtype=np.float32)
            p2 = np.asarray(p2, dtype=np.float32)

        sij = np.dot(sj, np.linalg.inv(si))  # 从i到j的视角刚体变换 , S 是世界坐标系到相机坐标系的变换
        Rij = sij[:3, :3]
        tij = sij[:3, -1]

        F = ... # 合成F
        np.save(Rij,tij,...) # 保存标签
```

产生Groundtruth的实现在create_label.py中实现，修改``dataset_dir = 'Angle30_60_dataset'``
即可产生和保存relative view label

## 代码功能介绍
### 1. 产生数据集的标签
blender渲染出来的数据集是没有相对姿态和Fundamental Matrix的标签的，因此需要自己根据绝对位姿计算出相对位姿。

在IDE环境中运行，工作路径设置为Project/code，然后运行create_label.py即可，运行成功应该可以看到绘制了极线图

**注意：create label用到的mathutils package在windows跑上会报错，因此需要在linux环境中运行**
```python
    # 修改start--------------------
    dataset_dir = 'SmallAngle_dataset'
    create_label = True 
    # create_label = False
    dataset_path = os.path.join(root_dir, dataset_dir)
    # 修改end-------------------
```
在主函数中修改上面的dataset_dir即可为不同数据集产生标签

### 2. 数据集的加载torch dataset
dataloader的代码都放在了Project/code/key_modules/mydataloader.py中

自己定义的Dataset类为MultiView dataset，默认__get_item__返回

```x,y=torch.cat(img1,img2),(F,pts1,pts2.T,Rt)```

### 3. net and loss
模型和loss设计都放在了Project/code/key_modules/net_and_loss.py中

根据params直接fetch得到net和loss实例

### 4. optimizer and scheduler
模型和loss设计都放在了Project/code/key_modules/optim.py中
根据params直接fetch得到optimizer和 scheduler

### 5. train demo
为了简单的演示整个训练的流程，我写了一个train_demo.py,可以参考该文件调用函数的API
```shell
cd Project/code
python train_demo.py -gpu=0,1,2  -exp=../experiments/base
```

###  6.评估网络模型性能
出于兼容性的考虑，我是把evaluate和train两个步骤分开写的。

evaluate网络性能的代码在Project/code/evaluate.py中

实际使用的时候，在一个experiment父级目录下(如exp1_learning_rate)，已经train了n个job，每个job都
有独立的文件夹，并且存在checkpoint.pth文件，即train的流程已经完成了，那么就可以直接运行evaluate_cmds.py开多线程在单个GPU上评估网络性能

并且生成markdown表格results.md文件
```shell
cd Project/code
python evalute_cmds.py -d=0 -parent=../experiments/exp1_learning_rate
```

