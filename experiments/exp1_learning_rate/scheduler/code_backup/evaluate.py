import json
import os
import cv2 as cv
import torch
from pylab import *
import matplotlib.pyplot as plt
import argparse

from mymodel import mynet, mydataloader
from create_label import computeF_with_cv
from abandoned import main
import utils
from mymodel.mynet import fetch_net_and_loss

evaluate_parser = argparse.ArgumentParser()
group = evaluate_parser.add_mutually_exclusive_group()
group.add_argument('-train', '--use_train_set', action='store_true',
                   help="Evaluate networks on train dataset")
group.add_argument('-test', '--use_test_set', action='store_true',
                   help="Evaluate networks on test dataset")

evaluate_parser.add_argument('--dataset', default=None,
                             help="Directory containing the dataset")
evaluate_parser.add_argument('-save', '--save_metric', action='store_true',
                             help="Save evaluate metrics")
evaluate_parser.add_argument('-exp', '--experiment_dir', default='../experiments/exp14_challenge_FreeView/dataset/dataset_FreeView_copy_dataset,FreeView_invariable_scene_dataset,SmallAngle_dataset,Angle30_60_dataset,MultiView_dataset',
                             type=str,
                             help='The directory to run a experiment and save results')
evaluate_parser.add_argument('-d', '--device', default='3', help='device id (i.e. 0 or 0,1 or cpu)')
evaluate_parser.add_argument('-rank', '--rank', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
evaluate_parser.add_argument('--seed',
                             default=230,
                             type=int,
                             help='seed for initializing training. ')

def GetPolarLine(FF, x, col, row):
    """
    绘制极线
    F:基础矩阵(极线公式l=F.dot(x))
    x:齐次坐标(3,1)向量
    col:图像的宽度
    row:图像的高度
    out：确定绘制线段的两个端点[x1,x2],[y1,y2]
    """
    Polarline = FF.dot(x)  # 计算极线的向量表达式[A,B,C]--->>Ax+By+C = 0
    # 找极线对应图像边缘的x,y坐标端点，向右为x正方向，向下为y方向正方向
    x1 = -Polarline[2] / Polarline[0]
    x2 = -(Polarline[2] + Polarline[1] * row) / Polarline[0]
    y1 = -Polarline[2] / Polarline[1]
    y2 = -(Polarline[2] + Polarline[0] * col) / Polarline[1]
    outx = []
    outy = []
    if 0 < x1 < col:
        outx.append(x1)
        outy.append(0)

    if 0 < x2 < col:
        outx.append(x2)
        outy.append(row)

    if 0 < y1 < row:
        outx.append(0)
        outy.append(y1)

    if 0 < y2 < row:
        outx.append(col)
        outy.append(y2)

    if len(outx) != 2:
        return None, None
        pass

        print(len(outx))
        print(x, FF)
        print(x1, x2, y1, y2)
        print('图像边长', col, row)
        raise ValueError('计算端点错误！返回点个数不等于2')

    return np.array(outx, dtype=np.float), np.array(outy, dtype=np.float)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def plot_raw_image_pair(img1, img2, title=None):
    """
    显示原图对
    :param img1:
    :param img2:
    :param title:
    :return:
    """
    img3 = np.column_stack((img1, img2))
    # b, g, r = cv.split(img3)
    # img3 = cv.merge([r, g, b])
    # img3 = rgb2gray(img3)
    plt.figure(figsize=(16, 8))
    plt.imshow(img3)
    if not title:
        plt.title('raw_image_pair')
    else:
        plt.title(title)
    # plt.axis('on')
    plt.axis('off')
    plt.show()


def plot_polarline(F_e, img1, img2, p1=None, p2=None, show_cv=False, title=None):
    """
    根据估计出的F，绘制对极线，并且绘制opencv-ransac的结果,p1 -> [shape(N,3),list]
    :param F_e:
    :param img1:
    :param img2:
    :param p1:
    :param p2:
    :param show_cv:
    :return:
    """

    # img3 = sift.appendimages(img1,img2)
    # img3 = vstack((img3, img3))

    img3 = np.column_stack((img1, img2))
    img3 = np.row_stack((img3, img3))
    # b, g, r = cv.split(img3)
    # img3 = cv.merge([r, g, b])
    # img3 = rgb2gray(img3)
    fig1 = plt.figure(figsize=(16, 8))
    plt.imshow(img3)
    cols1 = img1.shape[1]
    # print('Cols1: ',cols1)
    rows1 = img1.shape[0]

    if not isinstance(p1, np.ndarray):
        x1n, area1 = p1.copy()
        x2n, area2 = p2.copy()

        fig1.gca().add_patch(plt.Rectangle(*area1, linestyle='--', linewidth=3, edgecolor='r', facecolor='none'))
        print('area2: ', area2)
        area2 = area2.copy()
        area2[0] = area2[0][0] + cols1, area2[0][1]
        print('area2: ', area2)
        fig1.gca().add_patch(plt.Rectangle(*area2, linestyle='--', linewidth=3, edgecolor='r', facecolor='none'))
        F1, _, _ = computeF_with_cv(img1, img2)  # x1n->shape(N,3)
        x1n = x1n.T
        print(x1n.shape)
        x2n = x2n.T
        assert x1n is not None
    else:
        # F1, x1n, x2n = computeF_with_cv(img1, img2)  # x1n->shape(N,3)
        # x1n = x1n.T
        # x2n = x2n.T
        x1n = p1.T
        x2n = p2.T

    epi_threshold = 1

    # 估计出的F的结果
    # 下面的x1n的shape---->(3,N)
    # index =  range(x1n.shape[1])
    index = np.random.choice(np.arange(x1n.shape[1]), 10, replace=False)
    for i in index:
        # i就是ransac找出的good匹配点对的序号
        if (0 <= x1n[0][i] < cols1) and (0 <= x2n[0][i] < cols1) and (0 <= x1n[1][i] < rows1) and (
                0 <= x2n[1][i] < rows1):
            rgb = (np.random.randint(255) / 255, np.random.randint(255) / 255,
                   np.random.randint(255) / 255)  # 设置循环的rgb值，特征点和线的颜色相同

            plt.plot([x1n[0][i], x2n[0][i] + cols1], [x1n[1][i], x2n[1][i]], color=rgb, linewidth=2)  # 绘制匹配点连线
            plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]), s=np.pi * 5 ** 2,
                        c=[rgb], marker='x')

            # 在下方图中绘制极线
            xr, yr = GetPolarLine(F_e, x1n[:, i], cols1, rows1)  # 右图极线端点
            xl, yl = GetPolarLine(F_e.T, x2n[:, i], cols1, rows1)  # 左图极线端点

            # 根据局内局外点画不同颜色的线
            if xr is None or xl is None:
                # 如果极线在平面上都看不到了
                plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]) + rows1,
                            s=np.pi * 6 ** 2,
                            c='red', marker='x')
                continue

            if np.abs(x2n[:, i].reshape(1, 3).dot(F_e).dot(x1n[:, i].reshape(3, 1))) < epi_threshold:  # 局内点

                plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]) + rows1, s=np.pi * 6 ** 2,
                            c=[rgb], marker='o')
                plt.plot(xr + cols1, yr + rows1, color=rgb, linewidth=3)
                plt.plot(xl, yl + rows1, color=rgb, linewidth=3)
            else:  # 局外点
                plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]) + rows1, s=np.pi * 6 ** 2,
                            c='red', marker='s')
                plt.plot(xr + cols1, yr + rows1, '--m', c='red', linewidth=3)
                plt.plot(xl, yl + rows1, '--m', c='red', linewidth=3)
    if not title:
        plt.title('given_F')
    else:
        plt.title(title)
    # plt.axis('on')
    plt.axis('off')
    plt.show()

    if not show_cv:
        return

    plt.figure(figsize=(12, 6))
    plt.imshow(img3)

    # opencv的结果
    # 下面的x1n的shape---->(3,N)
    # for i in arange(x1n.shape[1]):
    for i in index:
        # i就是ransac找出的good匹配点对的序号
        if (0 <= x1n[0][i] < cols1) and (0 <= x2n[0][i] < cols1) and (0 <= x1n[1][i] < rows1) and (
                0 <= x2n[1][i] < rows1):
            plt.plot([x1n[0][i], x2n[0][i] + cols1], [x1n[1][i], x2n[1][i]], color='coral')  # 绘制匹配点连线
            plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]), s=np.pi * 3 ** 2,
                        c='red', marker='x')
            # 在下方图中绘制极线
            xr, yr = GetPolarLine(F1, x1n[:, i], cols1, rows1)  # 右图极线端点
            xl, yl = GetPolarLine(F1.T, x2n[:, i], cols1, rows1)  # 左图极线端点
            if xr is None or xl is None:
                plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]) + rows1, s=np.pi * 3 ** 2,
                            c='blue', marker='x')
                continue
            plt.scatter([x1n[0][i], x2n[0][i] + cols1], np.array([x1n[1][i], x2n[1][i]]) + rows1, s=np.pi * 3 ** 2,
                        c='red', marker='x')
            plt.plot(xr + cols1, yr + rows1, '-', linewidth=3)
            plt.plot(xl, yl + rows1, '-', linewidth=3)

    plt.title('opencv_result')
    plt.axis('off')
    plt.show()


def evaluate_fundamental(net, evaluate_dataset, experiment_dir=None, use_train_set=True, loss_module=None,
                         sample_num_to_show=1,
                         sample_num_to_evaluate=200, save_metric=False):
    assert experiment_dir and os.path.isdir(experiment_dir), 'No experiment_dir was found!'
    size_set = len(evaluate_dataset)
    loss_ransac = []
    loss_lmeds = []
    loss_gt = []
    loss_net = []

    ransac_statistic = []

    j = 0

    # for i in np.random.choice(np.arange(train_size, size_set), 20, replace=False): #在测试集上测试
    if loss_module:
        loss = loss_module()
    else:
        loss = mynet.loss_l1_l2()

    index = np.random.choice(np.arange(size_set), sample_num_to_evaluate, replace=False)  # 随机抽取样本测试

    show_index = np.random.choice(index,sample_num_to_show)

    # index = [1294,1789]
    min_loss = 10
    max_loss = 0
    for num, i in enumerate(index):  # 在训练集上测试

        img_pair, label = evaluate_dataset[i]  # img_pair--->(C,H,W)
        print("processing {}/{}: sample {}".format(num+1,sample_num_to_evaluate, i))

        img_pair = img_pair.reshape(1, *img_pair.shape).cuda(device=device)

        # numpy 格式参数
        Groundtruth = label[0].copy()
        p1, p2 = label[1:3]
        p1 = p1.T.copy()
        p2 = p2.copy()

        # 转为GPU tensor
        label = [torch.from_numpy(l).to(device, dtype=torch.float32) for l in label]
        rt = label[-1].reshape(1, -1)

        with torch.no_grad():
            if net.out_features == 7:
                # Y = net.evaluate_Rt_separately(img_pair, evaluate_R=False, Rt_label=rt)[0].cpu()
                Y = net(img_pair, Rt_label=rt)[0].cpu()
                # print('Evaluate R in RT ',end='')
                # print('Evaluate T in RT ', end='')
            else:
                Y = net(img_pair, Rt_label=rt)[0].cpu()
            Y = Y.numpy().reshape(3, 3)

            img1, img2 = evaluate_dataset.get_raw_imgs_with_label(i)[:2]
            F_ransac, pt1, pt2 = computeF_with_cv(img1, img2, match_threshold=0.2,
                                                  use_raw_feature_points=False)  # pt1， pt2 -> shape(N,3)
            F_lmeds, _, _ = computeF_with_cv(img1, img2, mode=cv.FM_LMEDS,
                                             match_threshold=0.2)  # pt1， pt2 -> shape(N,3)
            if not F_ransac.any() or not F_lmeds.any():
                print('Traditional methods failed to compute F! ')
                continue

            # F_lmeds, _ = cv.findFundamentalMat(p1,p2 ,method=cv.FM_8POINT) # 传统算法LMedS

            # 特征点的使用
            # x1n, _ = p1 # 老视频数据集
            # x2n, _ = p2
            # x1n = pt1 # 用Sift的特征点
            # x2n = pt2
            x1n = p1  # 用label中的特征点
            x2n = p2
            # if j < sample_num_to_show:
            if i in show_index:
                plot_polarline(F_ransac, img1, img2, p1=x1n, p2=x2n, title='Ransac')
                plot_polarline(F_lmeds, img1, img2, p1=x1n, p2=x2n, title='LMedS')
                plot_polarline(Y, img1, img2, p1=x1n, p2=x2n, title='MyNet')
                plot_polarline(Groundtruth, img1, img2, p1=x1n, p2=x2n, title='GroundTruth')

                # j += 1

            ransac_epi_per_point = np.abs(np.diagonal(x2n.dot(F_ransac.dot(x1n.T)))).mean()  # (200,) 求mean到每个特征点对
            loss_ransac.append(ransac_epi_per_point)

            lmeds_epi_per_point = np.abs(np.diagonal(x2n.dot(F_lmeds.dot(x1n.T)))).mean()
            loss_lmeds.append(lmeds_epi_per_point)

            gt_epi_per_point = np.abs(np.diagonal(x2n.dot(Groundtruth.dot(x1n.T)))).mean()
            loss_gt.append(gt_epi_per_point)

            net_epi_per_point = np.abs(np.diagonal(x2n.dot(Y.dot(x1n.T)))).mean()
            loss_net.append(net_epi_per_point)

            ransac_statistic.append(ransac_epi_per_point / net_epi_per_point)

            # 记录表现最好和最坏的样本
            if net_epi_per_point > max_loss:
                max_loss = net_epi_per_point
                max_loss_index = i
            if net_epi_per_point < min_loss:
                min_loss = net_epi_per_point
                min_loss_index = i

            # L1L2_ransac = loss(torch.from_numpy(F_ransac).float().cuda(device), label).cpu().numpy()
            # L1L2_lmeds = loss(torch.from_numpy(F_lmeds).float().cuda(device), label).cpu().numpy()
            # L1L2_Net = loss(torch.from_numpy(Y).float().cuda(device), label).cpu().numpy()
            # print("MATRIX DISTANCE >> Ransac:%.6f(%f)  LMedS:%.6f(%f)  Net:%f(1)" % (
            #     L1L2_ransac, L1L2_ransac / L1L2_Net, L1L2_lmeds, L1L2_lmeds / L1L2_Net, L1L2_Net))

    print('Experiment dir: ', experiment_dir)
    print('All evaluated samples count: {}'.format(sample_num_to_evaluate),
          ' on ' + ('train set' if use_train_set else 'test set'))
    print('Evaluation result>> Ransac:%.5f LMedS:%.5f Groundtruth:%.5f net:%.5f' % (
        mean(loss_ransac).__float__(), mean(loss_lmeds).__float__(), mean(loss_gt).__float__(),
        mean(loss_net).__float__()))
    print('Normalized Result>> Ransac:%.5f LMedS:%.5f Groundtruth:%.5f net:%.5f' % (
        sum(loss_ransac) / sum(loss_net).__float__(), sum(loss_lmeds) / sum(loss_net).__float__(),
        sum(loss_gt) / sum(loss_net).__float__(), sum(loss_net) / sum(loss_net).__float__()))

    print('Net worst performance on sample {}, loss: {}\nNet best performance on {}, loss: {}'.format(max_loss_index,
                                                                                                      max_loss,
                                                                                                      min_loss_index,
                                                                                                      min_loss))
    plt.figure()
    bin_num, _, _ = plt.hist(ransac_statistic, np.arange(60))
    assert len(ransac_statistic) > 0
    plt.title("Ransac/net relative epi loss distribution")
    plt.xlabel("Relative epi loss: |Ax0 + By0 + C|")
    plt.ylabel("sample count")
    plt.show()
    net_better_performance_rate = round(1 - bin_num[0] / len(ransac_statistic),3) # 四舍五入三位小数
    print('Rate of samples that net performs better than Ransac: ', net_better_performance_rate)

    if save_metric:
        metric = {}
        F_score = float(mean(loss_net))
        metric.update(dict(F_score=F_score, net_better_performance_rate=net_better_performance_rate))
        file_path = os.path.join(experiment_dir, 'metrics_train_set.json' if use_train_set else 'metrics_test_set.json')
        with open(file_path, 'w') as f:
            json.dump(metric, f, indent=4)

        print('===================Threads exit====================')


def evaluate_Rt(net, val_dataset, use_train_set=True):
    size_set = len(val_dataset)
    train_size = int(0.8 * size_set)
    # for i in np.random.choice(np.arange(train_size, size_set), 20, replace=False): #在测试集上测试
    loss = main.loss_Rt()
    j = 0
    if use_train_set:
        index = np.random.choice(np.arange(train_size), 20, replace=False)  # 在训练集上测试
    else:
        index = np.random.choice(np.arange(train_size, size_set), 20, replace=False)
    for i in index:

        img_pair, label = val_dataset[i]  # img_pair--->(C,H,W)
        print("processing sample {}".format(i))

        img_pair = img_pair.reshape(1, *img_pair.shape)
        with torch.no_grad():
            Y = net(img_pair).cpu()
        img1, img2, Rt = val_dataset.get_raw_imgs_with_label(i)
        print('Rt MATRIX loss: ', loss(Y, label))
        Y = torch.flatten(Y)
        print((label - Y).numpy())

        # 转为numpy格式方便画图
        Y = Y.numpy()
        label = label.numpy()
        if j < 1:
            plot_raw_image_pair(img1=img1, img2=img2,
                                title='Groundtruth: ' + str(label[:3]) + '---' + str(label[3:]) + '\nNet: ' + str(
                                    Y[:3]) + '---' + str(Y[3:]))
            j += 1





if __name__ == '__main__':
    torch.set_printoptions(10)
    # 解析命令行参数
    args = evaluate_parser.parse_args()
    experiment_dir = args.experiment_dir
    all_datasets_root_dir = '../blender_dataset'
    use_train_set = not args.use_test_set # 想要默认使用训练集
    device = int(args.device)

    # 读取训练参数文件
    json_path = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)  # dict file
    print('Evaluating experiment at {}'.format(experiment_dir))
    # 设置随机数
    # 设置随机数种子
    args.rank = device
    utils.setup_seed(args, params)

    if args.save_metric: # linux命令行中运行，保存结果文件
        sample_num_to_show = 0
        if use_train_set:
            sample_num_to_evaluate = 2000
        else:
            sample_num_to_evaluate = 1000
    else: # 终端运行，显示结果
        sample_num_to_show = 4
        if use_train_set:
            sample_num_to_evaluate = 20
        else:
            sample_num_to_evaluate = 10



    if not args.dataset:
        images_dir = params.dataset  # 在训练网络的数据集上进行测试
    else:
        images_dir = args.dataset  # 在额外指定的数据集上进行测试

    # 设置评估网络模型
    net, _ = fetch_net_and_loss(params=params,device=device,evaluate_mode=True)

        # 加载数据集
    train_set,test_set = mydataloader.fetch_concat_dataset(params=params)


    parallel_net = net
    parallel_net.cuda(device=device)

    if args.use_test_set:  # 用测试集
        whole_set = test_set
        checkpoint = torch.load(experiment_dir + os.sep + 'best_checkpoint.pth', map_location='cpu')
        print('Use the best net on test dataset!')
    else:
        whole_set = train_set
        checkpoint = torch.load(experiment_dir + os.sep + 'checkpoint.pth', map_location='cpu')
        print('Use the newest net!')

    net_dict = checkpoint['net_state_dict']
    new_dict={}

    # --------------------------------兼容性设计----------------------------

    for key,value in net_dict.items():
        if 'resNet' in key:
            new_dict.update({key.replace('resNet','backbone'):value})
        else:
            new_dict.update({key:value})
    parallel_net.load_state_dict(new_dict)

    # --------------------------------------------------------------------

    print(checkpoint['log_label'], checkpoint['trained_epoch'])
    parallel_net.eval()  # 改为评估模式，去掉drop层,修改bn层的作用方式
    loss_module = mynet.loss_reduce_epi
    evaluate_fundamental(net=parallel_net, evaluate_dataset=whole_set, experiment_dir=experiment_dir,
                         sample_num_to_evaluate=sample_num_to_evaluate,
                         use_train_set=use_train_set,
                         loss_module=loss_module, sample_num_to_show=sample_num_to_show, save_metric=args.save_metric)

    # 单独测试四元数标签与网络输出是否对的上
    if not args.save_metric:
        # 在pycharm终端运行时
        x,y = whole_set[0]
        y= [torch.from_numpy(label) for label in y]
        img1, img2, FF, p1, p2,Rt = whole_set.get_raw_imgs_with_label(0)
        loss = loss_module()
        with torch.no_grad():
            F,quat = parallel_net.cpu()(x)
            print(loss((F,quat), y))
        quat_gt = Rt.reshape(1, -1)
