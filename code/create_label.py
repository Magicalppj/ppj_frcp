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


def get_image_dir_and_files(root_dir=r'../dataset', dataset_dir='Original_dataset', file_name='sample_sequence.npy',
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
    return images_dir_path, image_files


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
    return np.array(m.to_euler()[:3])


def RotationMatrixToQuaternion(R):
    """
    将旋转矩阵转为欧拉角
    :param R:
    :return:
    """
    assert (isRotationMatrix(R))
    m = mathutils.Matrix(R)
    return np.array(m.to_quaternion()[:4])


def blender_create_MultiView_label(root_dir, images_dir='MultiView_dataset'):
    """
    根据现有的sample sequence（一般是通过原始数据集产生）
    """

    images_dir_path, image_files = get_image_dir_and_files(root_dir=root_dir, dataset_dir=images_dir,
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


    # 修改start--------------------
    dataset_dir = 'SmallAngle_dataset'
    create_label = True
    create_label = False
    dataset_path = os.path.join(root_dir, dataset_dir)
    # 修改end-------------------

    from key_modules import mydataloader

    # 生成fundamental matrix的标签
    if os.path.isdir(os.path.join(root_dir, dataset_dir, 'view1')):
        if create_label:
            blender_create_MultiView_label(root_dir=root_dir, images_dir=dataset_dir)
        whole_set = mydataloader.MultiViewDataset(root_dir=root_dir, dataset_dir=dataset_dir, transform=torchvision.transforms.ToTensor())
    else:
        raise ValueError('数据集目录下不存在view1和view2子目录')

    from evaluate import plot_polarline

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

    for i in np.concatenate([np.random.randint(0, len(whole_set), 5)]):
        img_left, img_right, F, p1, p2,_ = whole_set.get_raw_imgs_with_label(index=i)
        plot_polarline(F, img_left, img_right, p1, p2, show_cv=False, title='sample: ' + str(i))


    # save_multiview_samples(dataset_path=dataset_path,current_dataset=whole_set) # 保存图像对
