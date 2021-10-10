import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random
import mathutils
import json
import torchvision.transforms
from mpl_toolkits.mplot3d import Axes3D

# import mydataset
img2npy = lambda x: x.replace('jpg', 'npy').replace('png', 'npy')
check_item_in_range = lambda p, rmin, rmax: (p < rmax) * (p > rmin)
H, W = 512, 758


def get_label_file(img1, img2):
    """
    根据img1和img2的字符串获得label
    :param img1:
    :param img2:
    :return:
    """
    return img1.split('.')[0] + '_' + img2.split('.')[0] + '.npy'


def get_sample(img_left_path):
    """
    获取数据，包括图像，特征点，以及标签F
    :param img_left_path:
    :return:
    """
    img_right_path = img_left_path.replace('left', 'right')
    pts1_path = img2npy(img_left_path)
    pts2_path = img2npy(img_right_path)
    label_path = img2npy(img_left_path.replace('left', 'label'))

    img1 = cv.imread(img_left_path, 1)
    img2 = cv.imread(img_right_path, 1)
    pts1 = np.load(pts1_path)
    pts2 = np.load(pts2_path)
    label = np.load(label_path)

    return img1, img2, label, pts1, pts2


def get_images_path(root_dir, img_format):
    """
    获得双目图片对的地址字符串left,right以及文件名列表[xxx.png,xxx.png......]
    :param imgs_path_left: 左视角图片文件夹
    :param imgs_path_right: 右视角图片文件夹
    :param img_format: 检测图片的格式
    :return: imgs_path_left,imgs_path_right,image_files
    """
    imgs_path_left = os.path.join(root_dir, 'left')
    imgs_path_right = os.path.join(root_dir, 'right')
    save_path = os.path.join(root_dir, 'label')
    image_files = list(filter(lambda x: '.' + img_format in x, os.listdir(imgs_path_left)))  # 只检测目标格式的文件

    return imgs_path_left, imgs_path_right, save_path, image_files


def get_feature_points_sift(img_left, img_right):
    """
    使用sift检测并结合knn匹配特征点
    :param img_left: numpy array
    :param img_right:
    :return:
    """
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    if len(des1)<7 or len(des2)<7:
        return None,None
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # 根据Lowe的论文进行比率测试
    for i, (m, nn) in enumerate(matches):
        if m.distance < 0.8 * nn.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)  # .pt属性表示坐标
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.array(pts1)  # (N,2)
    pts2 = np.array(pts2)

    return pts1, pts2


def get_filtered_points(F, pts1, pts2, match_threshold=3):
    """
        使用已知的基础矩阵F对sift提取出来的特征点进行筛选，返回固定数量的特征点对
    :param F:
    :param pts1:
    :param pts2:
    :param match_threshold: epi的门限
    :return:
    """
    count_limit = 20
    flag = True
    homo_pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)  # shape->(N,3)
    homo_pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)
    line2 = np.dot(F, homo_pts1.T)  # line2 -> shape(3,N)
    numerator = np.abs(np.diag(homo_pts2.dot(line2)))  # 加了绝对值后的epi值
    denominator = np.linalg.norm(line2[:2], axis=0)  # 前两行为A,B
    # print(line2.shape,denominator.shape)
    pixel_distance = numerator / denominator  # shape -> (N,)
    mask = pixel_distance < match_threshold  # (N,), mask筛选局内点
    # print(np.sum(mask))
    if count_limit > np.sum(mask):
        flag = False
        return 0, 0, flag

    # homo_pts1 = homo_pts1[mask]
    # homo_pts2 = homo_pts2[mask]
    # index = np.random.choice(np.arange(homo_pts1.shape[0]), count_limit, replace=False) # 随机选取

    index = np.argsort(pixel_distance)[:count_limit]  # 按最小值排序选取
    return homo_pts1[index], homo_pts2[index], flag


def computeF_with_cv(img_left, img_right, mode=cv.FM_RANSAC, match_threshold=1, use_raw_feature_points=False):
    """
    输入np.array数据的左右图，使opencv计算并返回基础矩阵F和特征点对(N,3)
    :param img_left:
    :param img_right:
    :return:
    """
    pts1, pts2 = get_feature_points_sift(img_left=img_left, img_right=img_right)
    if  pts1 is None:
        return np.zeros((3, 3)),None,None

    if len(pts1)<1:
        return np.zeros((3, 3)),None,None

    F1, mask = cv.findFundamentalMat(pts1, pts2, mode, match_threshold)  # opencv实现
    if not isinstance(F1, np.ndarray):
        F1 = np.zeros((3, 3))
        homo_pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)  # shape->(N,3)
        homo_pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)
        return F1, homo_pts1, homo_pts2
    mask = mask.reshape(-1)
    # print(type(mask),pts1.shape)
    if use_raw_feature_points:
        homo_pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)  # shape->(N,3)
        homo_pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)
    else:
        homo_pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)[mask == 1]  # shape->(N,3)
        homo_pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)[mask == 1]
    return F1[:3], homo_pts1, homo_pts2  # 有时候返回值的F会是三个fundamental matrix，不符合标准


def computeF(pts1, pts2, mode=cv.FM_8POINT, match_threshold=1):
    """
    使用npy文件中的特征点计算基础矩阵
    :param pts1:
    :param pts2:
    :param mode:
    :param match_threshold:
    :return:
    """
    F1, mask = cv.findFundamentalMat(pts1, pts2, mode, match_threshold)  # opencv实现
    # homo_pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)  # shape->(N,3)
    # homo_pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)
    return F1, None, None
    pass


def get_sample_sequence(root_dir=r'../dataset', dataset_dir='Original_dataset', file_name='sample_sequence.npy',
                        recreate=False, size=3000):
    """
    生成数据集样本的采样序列，返回图片文件名列表以及（N，2）的索引序列
    :param root_dir:
    :param size:
    :return:
    """
    images_dir_path = os.path.join(root_dir, dataset_dir)
    if os.path.isdir(os.path.join(images_dir_path, 'view1')):
        print('Existing view1 and view2 dirs! ')
        view1_dataset_path = os.path.join(images_dir_path, 'view1')
        view2_dataset_path = os.path.join(images_dir_path, 'view2')
        all_files = list(
            set(os.listdir(view1_dataset_path)).intersection(set(os.listdir(view2_dataset_path))))
        image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
        image_files.sort()
    else:
        all_files = os.listdir(images_dir_path)
        image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
        image_files.sort()
    sample_sequence_path = os.path.join(root_dir, file_name)
    if not os.path.exists(sample_sequence_path):  # 如果不存在索引序列文件，就重新生成一个
        # print('Sample sequence not exist, will create a new sequence')
        # sample = lambda x: np.random.choice(x, 2, replace=False)
        # images_sequence = np.arange(len(image_files)).reshape(1, -1).repeat(size, axis=0)
        # sample_sequence = np.array(list(map(sample, images_sequence)))
        # np.save(sample_sequence_path, sample_sequence)
        pass
    else:
        sample_sequence = np.load(sample_sequence_path)
        n = sample_sequence.shape[0]
        print('Loading sample sequence, num of total samples:', n)

        if recreate:  # 如果数据有变化，且需要重新生成，那就重生生成一个
            print('Recreate a new sequence:', size)
            sample = lambda x: np.random.choice(x, 2, replace=False)
            images_sequence = np.arange(len(image_files)).reshape(1, -1).repeat(size, axis=0)
            sample_sequence = np.array(list(map(sample, images_sequence)))
            np.save(sample_sequence_path, sample_sequence)
    return images_dir_path, image_files, sample_sequence


def isRotationMatrix(R):
    """
    判断输入的矩阵是否是旋转矩阵
    :param R:
    :return:
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    # print("%.3e"%n,end="")
    # return n < 1e-6
    return n < 1e-5


def RotationMatrixToEulerAngles(R):
    """
    将旋转矩阵转为欧拉角
    :param R:
    :return:
    """
    assert (isRotationMatrix(R))
    m = mathutils.Matrix(R)
    # sy = math.sqrt(R[0, 0] ** 2 + R[2, 2] ** 2)
    #
    # singular = sy < 1e-6
    #
    # if not singular:
    #     x = math.atan2(R[2, 1], R[2, 2])
    #     y = math.atan2(-R[2, 0], sy)
    #     z = math.atan2(R[1, 0], R[0, 0])
    # else:
    #     x = math.atan2(-R[1, 2], R[1, 1])
    #     y = math.atan2(-R[2, 0], sy)
    #     z = 0

    return np.array(m.to_euler()[:3])


def RotationMatrixToQuaternion(R):
    """
    将旋转矩阵转为欧拉角
    :param R:
    :return:
    """
    assert (isRotationMatrix(R))
    m = mathutils.Matrix(R)
    # sy = math.sqrt(R[0, 0] ** 2 + R[2, 2] ** 2)
    #
    # singular = sy < 1e-6
    #
    # if not singular:
    #     x = math.atan2(R[2, 1], R[2, 2])
    #     y = math.atan2(-R[2, 0], sy)
    #     z = math.atan2(R[1, 0], R[0, 0])
    # else:
    #     x = math.atan2(-R[1, 2], R[1, 1])
    #     y = math.atan2(-R[2, 0], sy)
    #     z = 0

    return np.array(m.to_quaternion()[:4])


def cv_create_F_label(root_dir, match_threshold=1, size=3000, recreate=True):
    """根据存放双目图像路径的列表img_pair读取图像，并使用opencv进行求出F和特征点对，保存为.npy
    同时生成并保存sequence file
    """
    images_dir_path, image_files, sample_sequence = get_sample_sequence(root_dir=root_dir, size=size, recreate=recreate)
    label_dir_path = os.path.join(images_dir_path, 'label')
    if recreate:
        print('Clearing label dir! ')
        shutil.rmtree(label_dir_path)
        os.mkdir(label_dir_path)

    # 加载相机参数和旋转向量
    camera_matrix = np.load(root_dir + os.path.sep + 'camera_matrix.npy')
    if os.path.exists(root_dir + os.path.sep + 'r_matrixs.npy'):
        r_matrixs = np.load(root_dir + os.path.sep + 'r_matrixs.npy')
    else:
        r_matrixs = None
        rvecs = np.load(root_dir + r'/rvecs.npy')
    tvecs = np.load(root_dir + r'/tvecs.npy')
    # 要储存的参数
    n = 0
    num = sample_sequence.shape[0]
    sample_sequence = np.zeros_like(sample_sequence) - 1  # 设置为-1
    angles = np.zeros(num)
    Dists = np.zeros(num)

    # 样本的角度范围设置
    angle_start = 60
    angle_end = 120
    angle_step = 5

    # 准备采样函数
    sample = lambda x: np.random.choice(x, 2, replace=False)
    images_sequence = np.arange(len(image_files))
    print(len(image_files))
    check_repeat = lambda sample, sequence: np.isin(sample[1],
                                                    sequence[np.where(sequence[:, 0] == sample[0])]).any()  # 检测重复的函数
    while n < num:  # 循环直到生成3000组图片
        image_pair_sequence = sample(images_sequence)
        if check_repeat(image_pair_sequence, sample_sequence):  # 检测到重复，则跳过
            continue
        i, j = image_pair_sequence
        img1 = image_files[i]
        img2 = image_files[j]
        if r_matrixs is not None:
            Ri = r_matrixs[i]  # 直接取矩阵
            Rj = r_matrixs[j]
        else:
            Ri, _ = cv.Rodrigues(rvecs[i])  # 向量转旋转矩阵
            Rj, _ = cv.Rodrigues(rvecs[j])
        ti = tvecs[i]
        tj = tvecs[j]
        label_file = get_label_file(img1, img2)
        # img_left = np.asarray(cv.imread(os.path.join(dir_left, img_file), 1), dtype=np.uint8)
        # img_right = np.asarray(cv.imread(os.path.join(dir_right, img_file), 1), dtype=np.uint8)
        # F, p1, p2 = computeF_with_cv(img_left, img_right, cv.FM_RANSAC, match_threshold=match_threshold)
        img_item_path_left = os.path.join(images_dir_path, img1)
        img_item_path_right = os.path.join(images_dir_path, img2)
        # 用特征点的部分
        # pts1, _ = np.load(img2npy(img_item_path_left), allow_pickle=True)
        # pts2, _ = np.load(img2npy(img_item_path_right), allow_pickle=True)
        # F, p1, p2 = computeF(pts1, pts2, match_threshold=match_threshold)  # 8点法计算F
        # if F is None:
        #     print('Computing F Error! Removing:', n + 1)
        #     continue

        Rij = np.dot(Rj, Ri.T)  # Ri是世界坐标系转到i坐标系，Rij是从i坐标系转到j坐标系的旋转矩阵
        angle = RotationMatrixToEulerAngles(Rij) * 180 / np.pi
        roll = angle[2]

        if not (angle_start < np.abs(roll) < angle_end):
            continue
        angles[n] = roll  # 保存欧拉角平均值作为指标

        # tij = tj - Rij.dot(ti)  # tij也是从i到j的向量
        tij = Ri.dot(tj - ti)

        Dists[n] = np.linalg.norm(tij)  # 保存距离作为指标
        tx = np.array([[0, -tij[2], tij[1]],
                       [tij[2], 0, -tij[0]],
                       [-tij[1], tij[0], 0]], dtype=np.float)  # 向量叉乘的矩阵

        print("Processing:", n + 1, label_file)
        sample_sequence[n] = image_pair_sequence
        n += 1
        F = np.linalg.inv(camera_matrix).T.dot(tx).dot(Rij).dot(np.linalg.inv(camera_matrix))

        np.save(os.path.join(label_dir_path, label_file), F / F[-1, -1])

    # step = list(range(angle_start, angle_end, angle_step)) + list(
    #     range(-angle_end, -angle_start, angle_step))  # 生成区间，5度一区间
    # plt.figure()
    # fig, _, _ = plt.hist(Angles, np.arange(-angle_end, angle_end + 1, 5))  # fig为每个bar统计样本个数
    # plt.show()
    # threshold = np.min(fig[np.nonzero(fig)]).astype('int')  # 找到数量最小的门限，然后一刀切
    # print('Hist bar count threshold: ', threshold)
    # bar_choice = lambda x: np.random.choice(np.where((Angles < x + 5) & (Angles > x))[0],
    #                                         threshold, replace=False)  # 定义函数，从落到区间的元素集合中随机取门限个
    # sequence_index = np.concatenate(list(map(bar_choice, step)))  # map映射到每个区间，然后concat在一起组成一个index array
    # print(sequence_index.shape, np.unique(sequence_index).shape)
    # sequence_index = np.unique(sequence_index)
    # # sequence_index = np.arange(sample_sequence.shape[0])
    # plt.figure()
    # fig2, _, _ = plt.hist(Angles[sequence_index], np.arange(-angle_end, angle_end + 1, 5))
    # plt.show()
    sequence_index = np.arange(sample_sequence.shape[0])
    np.save(os.path.join(root_dir, 'sample_sequence.npy'), sample_sequence[sequence_index])
    np.save(os.path.join(root_dir, 'Angles.npy'), Angles[sequence_index])
    np.save(os.path.join(root_dir, 'Dists.npy'), Dists[sequence_index])


def blender_create_label_with_sequence(root_dir, dataset_dir='Original_dataset', match_threshold=3, num_of_each_bin=None, size=3000,
                                       recreate=True):
    """
    根据存放双目图像路径的列表img_pair读取图像，保存Fundamental matrix 和 feature points，保存为.npy
    同时生成并保存sequence file
    """

    # 样本的角度范围设置
    angle_start = 40
    angle_end = 120
    angle_step = 5
    _, bin_points, _ = plt.hist(np.zeros((100,)), np.arange(-angle_end, angle_end + 1, 5))  # fig为每个bar统计样本个数，获得区间左端点
    bins_counts = np.where((angle_start <= bin_points) * (bin_points < angle_end) + (-angle_end <= bin_points) * (
            bin_points < -angle_start))[0].shape[0]  # 要生成的区间个数
    if num_of_each_bin:
        size = bins_counts * num_of_each_bin
    print('Target size for dataset: ', size)

    images_dir_path, image_files, sample_sequence = get_sample_sequence(root_dir=root_dir, dataset_dir=dataset_dir,
                                                                        size=size, recreate=recreate)
    label_dir_path = os.path.join(images_dir_path, 'label')
    if recreate:
        if os.path.exists(label_dir_path):
            print('Clearing label dir! ')
            shutil.rmtree(label_dir_path)
        os.mkdir(label_dir_path)
    else:
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)

        source_path = os.path.join(root_dir, 'Original_dataset', 'label')

        if os.path.exists(source_path):
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, label_dir_path)
                    print(src_file)
        return

    # 加载相机参数
    camera_matrix = np.load(images_dir_path + os.path.sep + 'camera_matrix.npy')

    # 准备储存列表
    n = 0
    num = sample_sequence.shape[0]
    sample_sequence = np.zeros_like(sample_sequence) - 1  # 设置为-1
    Angles = np.zeros(num)
    Dists = np.zeros(num)

    images_total_num = len(image_files)  # 原图片的总数
    images_sequence = list(range(images_total_num))
    print(images_total_num)

    # 准备采样函数
    sample = lambda x: np.random.choice(x, 2, replace=False)  # 从序列中不放回地采取2个
    check_repeat = lambda sample, sequence: np.isin(sample[1],
                                                    sequence[np.where(sequence[:, 0] == sample[0])]).any()  # 检测重复的函数
    end_flag = False
    # while n < num:  # 循环直到生成num组图片
    while not end_flag:  # 循环直到生成num组图片
        '''
        # 旧方法，暴力循环产生数据
        image_pair_sequence = sample(images_sequence)
        if check_repeat(image_pair_sequence, sample_sequence):  # 检测到重复，则跳过
            continue
        '''

        i = random.choice(images_sequence)
        for step in range(-images_total_num + 1, images_total_num):
            if step == 0:  # 跳过0
                continue
            j = i + step
            if j < 0 or j > images_total_num - 1:  # j必须在下标允许的范围内
                continue

            img1_str = image_files[i]
            img2_str = image_files[j]
            with open(images_dir_path + os.sep + img1_str.replace('png', 'json'), 'r') as file:
                si, _, p1 = json.load(file)  # si 是从世界坐标转到相机坐标的外参矩阵
                si = np.asarray(si, dtype=np.float32)
                p1 = np.asarray(p1, dtype=np.float32)
            with open(images_dir_path + os.sep + img2_str.replace('png', 'json'), 'r') as file:
                sj, _, p2 = json.load(file)
                sj = np.asarray(sj, dtype=np.float32)
                p2 = np.asarray(p2, dtype=np.float32)

            label_file = get_label_file(img1_str, img2_str)

            # sij = np.dot(np.linalg.inv(sj), si)  # 从i到j的外参矩阵
            sij = np.dot(sj, np.linalg.inv(si))  # 从i到j的外参矩阵
            Rij = sij[:3, :3]
            tij = sij[:3, -1]
            angle = RotationMatrixToEulerAngles(Rij) * 180 / np.pi
            roll = angle[1]

            if not (angle_start < np.abs(roll) < angle_end):
                print('angle not suitable:{}, continue'.format(angle))
                continue

            counts, bin_points, _ = plt.hist(Angles,
                                             np.arange(-angle_end, angle_end + 1, angle_step))  # fig为每个bar统计样本个数
            target_angle_index = np.where(
                (angle_start <= bin_points) * (bin_points < angle_end) + (-angle_end <= bin_points) * (
                        bin_points < -angle_start))
            roll_bin_index = np.where(roll >= bin_points)[0][-1]
            if not (counts[target_angle_index] < num_of_each_bin).any():  # 退出检查要在bin数量检查之前，不然无法激活
                print('All angle bin counts satisfied!')
                end_flag = 1
                break
            if counts[roll_bin_index] > num_of_each_bin - 1:  # 设置每个bin的样本数量
                # print(roll,' bin counts already satisfied! Continue',counts[target_angle_index])
                print(roll, ' bin counts already satisfied! Continue')
                continue

            Angles[n] = roll  # 保存欧拉角平均值作为指标

            Dists[n] = np.linalg.norm(tij)  # 保存距离作为指标
            tx = np.array([[0, -tij[2], tij[1]], [tij[2], 0, -tij[0]], [-tij[1], tij[0], 0]],
                          dtype=np.float32)  # 向量叉乘的矩阵

            sample_sequence[n] = i, j  # 因为i是不带放回地随机产生，j是按顺序生成，所以一般来说不存在图片的重复
            F = np.linalg.inv(camera_matrix).T.dot(tx).dot(Rij).dot(np.linalg.inv(camera_matrix))
            F /= F[-1, -1]  # 提前归一化
            F = F.astype(np.float)  # 最终用相机参数得到的基础矩阵F

            # 下面使用F进行特征点提取，匹配和筛选
            # img_left = np.asarray(cv.imread(os.path.join(images_dir, img1_str), 1), dtype=np.uint8)
            # img_right = np.asarray(cv.imread(os.path.join(images_dir, img2_str), 1), dtype=np.uint8)
            # pts1, pts2 = get_feature_points_sift(img_left=img_left, img_right=img_right)
            # x1n, x2n, flag = get_filtered_points(pts1=pts1, pts2=pts2, F=F, match_threshold=match_threshold)
            x1n, x2n, flag = get_filtered_points(pts1=p1[:, :2], pts2=p2[:, :2], F=F, match_threshold=match_threshold)
            # flag = 1
            x1n = p1  # 全部保留世界坐标
            x2n = p2

            if not flag:  # 如果提取出来的特征点数量不对，那就跳过
                print('Not enough feature points, continue xxxxxxxxxxxxxxxxxxxxxxxxxx')
                continue

            bool_index = check_item_in_range(p1[:, 0], 0, W) * check_item_in_range(p1[:, 1], 0, H) * check_item_in_range(
                p2[:, 0], 0, W) * check_item_in_range(p2[:, 1], 0, H)
            index = np.squeeze(bool_index)
            print('Sum index:{}, total index:{}'.format(sum(index), len(index)))
            np.save(os.path.join(label_dir_path, 'pindex_' + label_file), index)  # 保存特征点points_file文件
            np.save(os.path.join(label_dir_path, 'points_' + label_file), [x1n, x2n])  # 保存特征点points_file文件
            print("Processing:", n + 1, label_file, '\tfeature points:', x1n.shape,
                  'Roll angle: ', roll)

            np.save(os.path.join(label_dir_path, label_file), F)  # 保存gt

            n += 1
            if n > num - 1:
                break

        images_sequence.remove(i)
        if len(images_sequence) < 1 or end_flag:  # 当前数据集原图片数目不够产生那么多样本，则直接跳出循环
            sample_sequence = sample_sequence[:n]  # 把预先准备的数据集大小改变一下
            Angles = Angles[:n]
            Dists = Dists[:n]
            break
    '''
    step = list(range(angle_start, angle_end, angle_step)) + list(
        range(-angle_end, -angle_start, angle_step))  # 生成区间，5度一区间
    plt.figure()
    fig, _, _ = plt.hist(Angles, np.arange(-angle_end, angle_end + 1, 5))  # fig为每个bar统计样本个数
    plt.show()
    threshold = np.min(fig[np.nonzero(fig)]).astype('int')  # 找到数量最小的门限，然后一刀切
    print('Hist bar count threshold: ', threshold)
    bar_choice = lambda x: np.random.choice(np.where((Angles < x + 5) & (Angles > x))[0],
                                            threshold, replace=False)  # 定义函数，从落到区间的元素集合中随机取门限个
    sequence_index = np.concatenate(list(map(bar_choice, step)))  # map映射到每个区间，然后concat在一起组成一个index array
    print(sequence_index.shape, np.unique(sequence_index).shape)
    sequence_index = np.unique(sequence_index)
    plt.figure()
    fig2, _, _ = plt.hist(Angles[sequence_index], np.arange(-angle_end, angle_end + 1, 5))
    plt.show()
    '''
    sequence_index = np.random.permutation(sample_sequence.shape[0])  # 全部保留并且打乱顺序

    np.save(os.path.join(root_dir, 'sample_sequence.npy'), sample_sequence[sequence_index])
    np.save(os.path.join(root_dir, 'Angles.npy'), Angles[sequence_index])
    np.save(os.path.join(root_dir, 'Dists.npy'), Dists[sequence_index])


def blender_create_Rt_label(root_dir=r'../blender_dataset', images_dir='Original_dataset', num_of_each_bin=None,
                            size=3000, recreate_sequence=False):
    """根据存放双目图像路径的列表img_pair读取图像，并使用opencv进行求出F和特征点对，保存为.npy
    同时生成并保存sequence file
    """

    if os.path.isdir(os.path.join(root_dir, images_dir, 'view1')):
        images_dir_path, image_files, _ = get_sample_sequence(root_dir=root_dir, dataset_dir=images_dir,
                                                              )
        label_dir_path = os.path.join(root_dir, images_dir, 'Rt_label')
        if os.path.isdir(label_dir_path):
            print('Clearing label dir! ', label_dir_path)
            shutil.rmtree(label_dir_path)
        os.mkdir(label_dir_path)
        # 加载相机参数
        num = len(image_files)
        Euler_Angles = np.zeros((num, 3))
        Dists = np.zeros(num)
        for n, image_file_name in enumerate(image_files):
            img1_str = image_file_name
            img2_str = image_file_name
            with open(os.path.join(root_dir, images_dir, 'view1', img1_str.replace('png', 'json')), 'r') as file:
                si, _, p1 = json.load(file)  # si 是从世界坐标转到相机坐标的外参矩阵
                si = np.asarray(si, dtype=np.float32)
                p1 = np.asarray(p1, dtype=np.float32)
            with open(os.path.join(root_dir, images_dir, 'view2', img2_str.replace('png', 'json')), 'r') as file:
                sj, _, p2 = json.load(file)
                sj = np.asarray(sj, dtype=np.float32)
                p2 = np.asarray(p2, dtype=np.float32)

            label_file_name = 'Rt_' + img2npy(image_file_name)

            sij = np.dot(sj, np.linalg.inv(si))  # 从i到j的外参矩阵 , S 是世界坐标系到相机坐标系的变换
            Rij = sij[:3, :3]
            tij = sij[:3, -1]

            angle = RotationMatrixToEulerAngles(Rij) * 180 / np.pi
            Euler_Angles[n] = angle  # 保存欧拉角平均值作为指标
            Dists[n] = np.linalg.norm(tij)  # 保存距离作为指标
            normalized_tij = tij / Dists[n]

            print("Processing:", n + 1, label_file_name)
            quaternion = RotationMatrixToQuaternion(Rij)
            bool_index = check_item_in_range(p1[:, 0], 0, W) * check_item_in_range(p1[:, 1], 0, H) * check_item_in_range(
                p2[:, 0], 0, W) * check_item_in_range(p2[:, 1], 0, H)
            index = np.random.shuffle(np.where(bool_index)[0])[:50]
            assert len(index) == 50
            print('Sum index:{}, total index:{}'.format(sum(index), len(index)))
            np.save(os.path.join(label_dir_path, 'points_' + label_file_name),
                    [p1[index], p2[index]])  # 保存特征点points_file文件
            np.save(os.path.join(label_dir_path, label_file_name),
                    np.concatenate([quaternion, normalized_tij], axis=0))  # 保存rt

            n += 1
        np.save(os.path.join(root_dir, images_dir, 'Rt_Angles.npy'), Euler_Angles)
        np.save(os.path.join(root_dir, images_dir, 'Rt_Dists.npy'), Dists)
    else:
        if recreate_sequence:
            # 样本的角度范围设置
            angle_start = 40
            angle_end = 120
            angle_step = 5
            _, bin_points, _ = plt.hist(np.zeros((100,)),
                                        np.arange(-angle_end, angle_end + 1, angle_step))  # fig为每个bar统计样本个数，获得区间左端点
            bins_counts = \
                np.where((angle_start <= bin_points) * (bin_points < angle_end) + (-angle_end <= bin_points) * (
                        bin_points < -angle_start))[0].shape[0]  # 要生成的区间个数
            if not num_of_each_bin:
                num_of_each_bin = int(size / bins_counts)
            size = bins_counts * num_of_each_bin
            print('Target size for Rt dataset: ', size)
            images_dir_path, image_files, sample_sequence = get_sample_sequence(root_dir=root_dir,
                                                                                dataset_dir=images_dir,
                                                                                file_name='Rt_sample_sequence.npy',
                                                                                size=size,
                                                                                recreate=recreate_sequence)
        else:
            # 读fundamental数据集的角度样本
            images_dir_path, image_files, sample_sequence = get_sample_sequence(root_dir=root_dir,
                                                                                dataset_dir=images_dir,
                                                                                file_name='sample_sequence.npy',
                                                                                size=size,
                                                                                recreate=recreate_sequence)
        label_dir_path = os.path.join(root_dir, images_dir, 'Rt_label')  #
        if os.path.exists(label_dir_path):
            print('Clearing label dir! ', label_dir_path)
            shutil.rmtree(label_dir_path)
        os.mkdir(label_dir_path)

        # 准备储存列表
        n = 0
        num = sample_sequence.shape[0]
        Angles = np.zeros(num)
        Dists = np.zeros(num)

        images_total_num = len(image_files)  # 原图片的总数
        images_sequence = list(range(images_total_num))
        print(images_total_num)

        if recreate_sequence:  # 重新随机产生样本
            while n < num:  # 循环直到生成num组图片
                i = random.choice(images_sequence)
                for step in range(-images_total_num + 1, images_total_num):
                    if step == 0:  # 跳过0
                        continue
                    j = i + step
                    if j < 0 or j > images_total_num - 1:  # j必须在下标允许的范围内
                        continue

                    img1_str = image_files[i]
                    img2_str = image_files[j]

                    with open(images_dir_path + os.sep + img1_str.replace('png', 'json'), 'r') as file:
                        si, _, p1 = json.load(file)  # si 是从世界坐标转到相机坐标的外参矩阵
                        si = np.asarray(si, dtype=np.float32)
                        p1 = np.asarray(p1, dtype=np.float32)
                    with open(images_dir_path + os.sep + img2_str.replace('png', 'json'), 'r') as file:
                        sj, _, p2 = json.load(file)
                        sj = np.asarray(sj, dtype=np.float32)
                        p2 = np.asarray(p2, dtype=np.float32)

                    label_file_name = get_label_file(img1_str, img2_str)

                    # sij = np.dot(np.linalg.inv(sj), si)  # 从i到j的外参矩阵
                    sij = np.dot(sj, np.linalg.inv(si))  # 从i到j的外参矩阵
                    Rij = sij[:3, :3]
                    tij = sij[:3, -1]
                    angle = RotationMatrixToEulerAngles(Rij) * 180 / np.pi
                    roll = angle[1]

                    if not (angle_start < np.abs(roll) < angle_end):
                        print('angle not suitable:{}, continue'.format(angle))
                        continue
                    quaternion = RotationMatrixToQuaternion(Rij)
                    Angles[n] = roll  # 保存欧拉角平均值作为指标
                    Dists[n] = np.linalg.norm(tij)  # 保存距离作为指标
                    sample_sequence[n] = i, j
                    print("Processing:", n + 1, label_file_name)
                    bool_index = check_item_in_range(p1[:, 0], 0, W) * check_item_in_range(p1[:, 1], 0,
                                                                                      H) * check_item_in_range(p2[:, 0],
                                                                                                               0,
                                                                                                               W) * check_item_in_range(
                        p2[:, 1], 0, H)
                    index = np.random.shuffle(np.where(bool_index)[0])[:50]
                    assert len(index) == 50
                    print('Sum index:{}, total index:{}'.format(sum(index), len(index)))
                    np.save(os.path.join(label_dir_path, 'points_' + label_file_name),
                            [p1[index], p2[index]])  # 保存特征点points_file文件
                    np.save(os.path.join(label_dir_path, 'Rt_' + label_file_name),
                            np.concatenate([quaternion, tij], axis=0))  # 保存rt

                    n += 1
                    if n > num - 1:
                        break

                images_sequence.remove(i)
                if len(images_sequence) < 1:  # 当前数据集原图片数目不够产生那么多样本，则直接跳出循环
                    sample_sequence = sample_sequence[:n]  # 把预先准备的数据集大小改变一下
                    Angles = Angles[:n]
                    Dists = Dists[:n]
                    break
            sequence_index = np.random.permutation(sample_sequence.shape[0])  # 打乱顺序随机排列样本
        else:  # 直接读现有的fundamental dataset sequence

            while n < num:  # 循环直到生成num组图片
                i, j = sample_sequence[n]
                img1_str = image_files[i]
                img2_str = image_files[j]

                with open(images_dir_path + os.sep + img1_str.replace('png', 'json'), 'r') as file:
                    si, _, p1 = json.load(file)  # si 是从世界坐标转到相机坐标的外参矩阵
                    si = np.asarray(si, dtype=np.float32)
                    p1 = np.asarray(p1, dtype=np.float32)
                with open(images_dir_path + os.sep + img2_str.replace('png', 'json'), 'r') as file:
                    sj, _, p2 = json.load(file)
                    sj = np.asarray(sj, dtype=np.float32)
                    p2 = np.asarray(p2, dtype=np.float32)

                label_file_name = get_label_file(img1_str, img2_str)

                # sij = np.dot(np.linalg.inv(sj), si)  # 从i到j的外参矩阵,Rt 是相机坐标到世界坐标
                sij = np.dot(sj, np.linalg.inv(si))  # 从i到j的外参矩阵，Rt是世界坐标到相机坐标
                Rij = sij[:3, :3]
                tij = sij[:3, -1]
                angle = RotationMatrixToEulerAngles(Rij) * 180 / np.pi
                roll = angle[1]
                Angles[n] = roll  # 保存欧拉角平均值作为指标
                Dists[n] = np.linalg.norm(tij)  # 保存距离作为指标

                print("Processing:", n + 1, label_file_name)
                quaternion = RotationMatrixToQuaternion(Rij)
                bool_index = check_item_in_range(p1[:, 0], 0, W) * check_item_in_range(p1[:, 1], 0, H) * check_item_in_range(
                    p2[:, 0], 0, W) * check_item_in_range(p2[:, 1], 0, H)
                index = np.random.shuffle(np.where(bool_index)[0])[:50]
                assert len(index) == 50
                print('Sum index:{}, total index:{}'.format(sum(index), len(index)))
                np.save(os.path.join(label_dir_path, 'points_' + label_file_name),
                        [p1[index], p2[index]])  # 保存特征点points_file文件
                np.save(os.path.join(label_dir_path, 'Rt_' + label_file_name),
                        np.concatenate([quaternion, tij], axis=0))  # 保存rt

                n += 1
            sequence_index = np.arange(sample_sequence.shape[0])  # 全部按顺序保留

        np.save(os.path.join(root_dir, 'Rt_sample_sequence.npy'), sample_sequence[sequence_index])
        np.save(os.path.join(root_dir, 'Rt_Angles.npy'), Angles[sequence_index])
        np.save(os.path.join(root_dir, 'Rt_Dists.npy'), Dists[sequence_index])


def blender_create_MultiView_label(root_dir, images_dir='MultiView_dataset'):
    """
    根据现有的sample sequence（一般是通过原始数据集产生）
    """

    images_dir_path, image_files, sample_sequence = get_sample_sequence(root_dir=root_dir, dataset_dir=images_dir,
                                                                        )
    label_dir_path = os.path.join(root_dir, images_dir, 'label')
    if os.path.isdir(label_dir_path):
        print('Clearing label dir! ', label_dir_path)
        shutil.rmtree(label_dir_path)
    os.mkdir(label_dir_path)

    Rt_label_dir_path = os.path.join(root_dir, images_dir, 'Rt_label')
    if os.path.isdir(Rt_label_dir_path):
        print('Clearing Rt_label dir! ', Rt_label_dir_path)
        shutil.rmtree(Rt_label_dir_path)
    os.mkdir(Rt_label_dir_path)

    # 加载相机参数
    camera_matrix = np.load(os.path.join(root_dir, images_dir, 'camera_matrix.npy'))
    num = len(image_files)
    Euler_Angles = np.zeros((num, 3))
    Dists = np.zeros(num)
    for n, image_file_name in enumerate(image_files):
        img1_str = image_file_name
        img2_str = image_file_name
        with open(os.path.join(root_dir, images_dir, 'view1', img1_str.replace('png', 'json')), 'r') as file:
            si, _, p1 = json.load(file)  # si 是从世界坐标转到相机坐标的外参矩阵
            si = np.asarray(si, dtype=np.float32)
            p1 = np.asarray(p1, dtype=np.float32)
        with open(os.path.join(root_dir, images_dir, 'view2', img2_str.replace('png', 'json')), 'r') as file:
            sj, _, p2 = json.load(file)
            sj = np.asarray(sj, dtype=np.float32)
            p2 = np.asarray(p2, dtype=np.float32)

        label_file_name = img2npy(image_file_name)
        Rt_label_file_name = 'Rt_'+img2npy(image_file_name)

        sij = np.dot(sj, np.linalg.inv(si))  # 从i到j的外参矩阵 , S 是世界坐标系到相机坐标系的变换
        Rij = sij[:3, :3]
        tij = sij[:3, -1]

        tx = np.array([[0, -tij[2], tij[1]], [tij[2], 0, -tij[0]], [-tij[1], tij[0], 0]],
                      dtype=np.float32)  # 向量叉乘的矩阵
        angle = RotationMatrixToEulerAngles(Rij) * 180 / np.pi
        Euler_Angles[n] = angle
        Dists[n] = np.linalg.norm(tij)

        normalized_tij = tij / Dists[n] # translation
        quaternion = RotationMatrixToQuaternion(Rij) # quaternion

        F = np.linalg.inv(camera_matrix).T.dot(tx).dot(Rij).dot(np.linalg.inv(camera_matrix))
        F /= F[-1, -1]  # 提前归一化
        F = F.astype(np.float)  # 最终用相机参数得到的基础矩阵F
        # bool_index = check_item_in_range(p1[:, 0], 0, W) * check_item_in_range(p1[:, 1], 0, H) * check_item_in_range(
        #     p2[:, 0], 0, W) * check_item_in_range(p2[:, 1], 0, H)

        # index = np.where(bool_index)[0]
        index = np.arange(p1.shape[0])
        print(index.shape)
        np.random.shuffle(index)
        index=index[:50]
        assert len(index)==50,'{} not enough feature points'.format(os.path.join(root_dir, images_dir, 'view1', img1_str.replace('png', 'json')))
        np.save(os.path.join(label_dir_path, 'points_' + label_file_name), [p1[index], p2[index]])  # 保存特征点points_file文件
        np.save(os.path.join(label_dir_path, label_file_name), F)  # 保存Fundamental matrix

        np.save(os.path.join(Rt_label_dir_path, 'points_' + Rt_label_file_name),
                [p1[index], p2[index]])  # 保存特征点points_file文件
        np.save(os.path.join(Rt_label_dir_path, Rt_label_file_name),
                np.concatenate([quaternion, normalized_tij], axis=0))  # 保存rt


        print("Processing:", n + 1, label_file_name)
    np.save(os.path.join(root_dir, images_dir, 'Angles.npy'), Euler_Angles)
    np.save(os.path.join(root_dir, images_dir, 'Dists.npy'), Dists)

def save_multiview_samples(dataset_path, current_dataset=None, indexes=None):
    if not indexes:
        indexes = np.random.choice(np.arange(len(current_dataset)), 10)

    sample_dir = 'samples'
    save_dir_path = os.path.join(dataset_path, sample_dir)
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)
        print('Making dir for saving: ', save_dir_path)
    for n in indexes:
        print('Saving sample ', n)
        parameters = current_dataset.get_raw_imgs(n)
        img1, img2 = parameters[:2]
        cat_img = cv.cvtColor(np.concatenate([img1, img2], axis=1), cv.COLOR_BGR2RGB)
        sample_name = 'sample_' + str(n).zfill(6) + '.png'
        cv.imencode('.png', cat_img, )[1].tofile(os.path.join(save_dir_path, sample_name))


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: .4f}'.format})
    root_dir = r'../blender_dataset'
    # 修改--------------------
    dataset_dir = 'MultiView_dataset'
    create_label = True
    create_label = False
    dataset_path = os.path.join(root_dir, dataset_dir)
    # 修改-------------------

    from mymodel import mydataloader

    # 生成fundamental matrix的标签
    if os.path.isdir(os.path.join(root_dir, dataset_dir, 'view1')):
        if create_label:
            blender_create_MultiView_label(root_dir=root_dir, images_dir=dataset_dir)
        whole_set = mydataloader.MultiViewDataset(root_dir=root_dir, dataset_dir=dataset_dir,transform=torchvision.transforms.ToTensor())
    else:
        if create_label:
            blender_create_label_with_sequence(root_dir=root_dir, dataset_dir=dataset_dir, num_of_each_bin=200, size=6000,
                                               recreate=False,
                                               match_threshold=3)
        whole_set = mydataloader.FundamentalDataset(root_dir=root_dir, dataset_dir=dataset_dir,transform=torchvision.transforms.ToTensor())


    from evaluate import plot_polarline, plot_raw_image_pair

    # Angles = np.load(root_dir + os.sep + 'Angles.npy')
    Angles = np.load(os.path.join(root_dir, dataset_dir, 'Angles.npy'))
    fig = plt.figure()
    # fig = plt.hist(Angles[:,1], np.arange(-165, 170, 5))
    ax = Axes3D(fig)
    ax.scatter(Angles[:, 0], Angles[:, 1], Angles[:, 2])
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Yaw')
    ax.set_zlabel('Roll')
    plt.title(r'Euler Angles distribution of dataset')
    # plt.xlabel(r'angle/°')
    # plt.ylabel(r'Sample number')
    plt.show()

    Dists = np.load(os.path.join(root_dir, dataset_dir, 'Dists.npy'))
    plt.figure()
    plt.hist(Dists)
    plt.title(r'Dists')
    plt.xlabel(r'Distance')
    plt.ylabel(r'Sample number')
    plt.show()

    # for i in range(900,905):
    for i in np.concatenate([np.random.randint(0, len(whole_set), 5)]):
        angle = Angles[i]
        dist = Dists[i]

        img_left, img_right, F, p1, p2,_ = whole_set.get_raw_imgs_with_label(index=i)
        plot_polarline(F, img_left, img_right, p1, p2, show_cv=False, title='sample: ' + str(i))

    save_multiview_samples(dataset_path=dataset_path,current_dataset=whole_set)
