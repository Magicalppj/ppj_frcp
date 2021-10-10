##处理图片，去畸变
import os
import random

import numpy as np
import cv2 as cv
import glob


def get_undistorted_images(images_path=r'../dataset/Original_dataset'):
    """
    生成去畸变后的图像，并且删除原图像，同时保存角点npy文件
    :param images_path:
    :return:
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,5,0)
    objp = np.zeros((7 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:7].T.reshape(-1, 2)
    # objp[:, :2] = np.mgrid[0:8, 0:7].T.reshape(-1, 2) * 30 #单位假设为30mm一格
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # images = glob.glob('*.jpg')

    all_files = os.listdir(images_path)
    image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
    image_files.sort()
    limit = 200
    if len(image_files) > limit:
        n = limit
    else:
        n = len(image_files)
    print('Total n:', n)
    for i, file in enumerate(random.sample(image_files, n )):
        fname = os.path.join(images_path, file)
        img = cv.imread(fname, 1)
        if img is None:
            print('Error! Read image failed', fname)
            return
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8, 7), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print('Check', i + 1, file)
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            # cv.drawChessboardCorners(img, (8, 7), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(200)
        else:
            print('Removing:', i + 1, file)
            os.remove(fname)
    # cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(type(rvecs), len(rvecs))
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    np.save(r'../dataset/distorted_camera_matrix.npy', mtx)
    np.save(r'../dataset/distortion_vector.npy', dist)
    np.save(r'../dataset/camera_matrix.npy', newcameramtx)
    print('Saving camera intrinsic matrix and distortion vector! ')

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    all_files = os.listdir(images_path)
    image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
    image_files.sort()

    # rvecs = np.array(rvecs)
    # tvecs = np.array(tvecs)
    # flag = np.ones(rvecs.shape[0])
    rvecs = []
    tvecs = []

    for i, file in enumerate(image_files):
        fname = os.path.join(images_path, file)
        img = cv.imread(fname, 1)
        # 先solvePNP计算出当前帧的R，t，再去畸变保存角点
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (8, 7), None)
        if ret is not True:
            # 如果检测角点失败，则删除
            print('Removing', i + 1, file)
            os.remove(fname)
            continue
        ret, rvec, tvec, _ = cv.solvePnPRansac(objp, corners, mtx, dist)

        if ret == True:  # 如果求解出了R，t
            # 保存R，t
            rvecs.append(rvec)
            tvecs.append(tvec)
            # undistort
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (8, 7), None)
            if ret is not True:
                # 如果检测角点失败，则删除
                print('Removing', i + 1, file)
                os.remove(fname)
                continue
            # If found, add object points, image points (after refining them)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners2 = corners2.reshape(corners2.shape[0], corners2.shape[2])
            print('Rectifying', i + 1, file, dst.shape)
            cv.imwrite(fname, dst)
            x_max, y_max = np.max(corners2, axis=0).astype('int') + 1
            x_min, y_min = np.min(corners2, axis=0).astype('int')
            np.save(fname.replace('.jpg', '.npy'), [np.concatenate((corners2, np.ones((corners2.shape[0], 1))), axis=1),
                                                    [(x_min, y_min), x_max - x_min,
                                                     y_max - y_min]])  # (N,2)->(N,3)转齐次坐标,同时保存area
        else:
            print('Removing', i + 1, file)
            # flag[i]=0
            os.remove(fname)
    # rvecs = rvecs[flag==1] #修正rvecs
    # tvecs = tvecs[flag==1]
    np.save(r'../dataset/rvecs.npy', rvecs)
    np.save(r'../dataset/tvecs.npy', tvecs)
    print('Saving rvecs and tvecs! ')
    all_files = os.listdir(images_path)
    image_files = list(filter(lambda x: '.png' in x or '.jpg' in x, all_files))
    print('Total images num:', len(image_files))


if __name__ == '__main__':
    # get_undistorted_images()
    # test_img = r'../dataset/Original_dataset/0000006.jpg'
    # img = cv.imread(test_img, 1)
    # a = np.load(test_img.replace('.jpg', '.npy'))
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # # plt.axis([0, 1196, 0, 643])
    # plt.imshow(img)
    # plt.scatter(a[:, 0], a[:, 1], c='red')
    # plt.show()
    # pass
    new_camera_mtx = np.load(r'../dataset/camera_matrix.npy')
    mtx = np.load(r'../dataset/distorted_camera_matrix.npy')
    dist = np.load(r'../dataset/distortion_vector.npy')
    print('new_camera_mtx:\n',new_camera_mtx)
    print('mtx:\n',mtx)
    print('dist:\n',dist)