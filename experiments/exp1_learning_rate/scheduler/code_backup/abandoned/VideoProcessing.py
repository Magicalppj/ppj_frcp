import cv2
import os
import time
import numpy as np
import shutil

from code.abandoned.ImageProcessing import get_undistorted_images

img2npy = lambda x: x.replace('jpg', 'npy').replace('png', 'npy')

def video_sampling(video_path, target_path,interval=2):
    if not os.path.exists(video_path):
        print("Path does not exist!")
        return

    if os.path.isfile(video_path):
        video_list = [video_path]
    else:
        video_list = os.listdir(video_path)

    # 保存图片的帧率间隔
    i = 0
    j = 0
    for index, video_name in enumerate(video_list):
        if len(video_list)>1:
            video_path_ = os.path.join(video_path, video_name)
        else:
            video_path_ = video_path

        # 开始读视频
        videoCapture = cv2.VideoCapture(video_path_)
        print("正在处理第{}个视频，总共{}个视频".format(index + 1, len(video_list)))

        while True:
            success, frame = videoCapture.read()
            i += 1
            if not success:
                print('video is all read')
                break

            if (i % interval == 0):
                # 保存图片
                j += 1
                pic_name = str(j).zfill(6) + '.jpg'
                # cv2.imwrite(os.path.join(target_path, pic_name), frame)
                cv2.imencode('.jpg', frame)[1].tofile(os.path.join(target_path, pic_name))
                print('image of %s is saved' % (pic_name))


        videoCapture.release()
        time.sleep(5)

def copy_items(images_path, target_path):
    """
    把jpg图片和npy文件从一个文件夹复制到另一个文件夹
    :param images_path:
    :param target_path:
    :return:
    """
    for root, dirs, files in os.walk(images_path):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.npy'):  # 若文件名结尾是以jpg结尾，则复制到新文件夹
                list = (os.path.join(root, name))  # list是jpg文件的全路径
                shutil.copy(list, target_path)  # 将jpg文件复制到新文件夹
    print('Copy completed')
    pass

def rename_items(images_path):
    """
    对指定目录文件夹下的jpg和npy文件一对一对地重新随机命名
    :param images_path:
    :return:
    """
    files = list(filter(lambda x:'.jpg' in x,os.listdir(images_path)))
    new_names =np.arange(len(files)) + 1
    np.random.shuffle(new_names)
    for i,file in enumerate(files):  # i是path目录下的文件名
        oldname = os.path.join(images_path, file)  # oldname是原文件路径
        newname = os.path.join(images_path, 'temp'+file)  # newname是新文件名，路径+新文件名
        os.rename(oldname, newname)  # 改名字
        os.rename(img2npy(oldname), img2npy(newname))  # 改名字
        # print(oldname, '======>', newname)
    for i,file in enumerate(files):  # i是path目录下的文件名
        oldname = os.path.join(images_path, 'temp'+file)  # oldname是原文件路径
        newname = os.path.join(images_path, str(new_names[i]) + '.jpg')  # newname是新文件名，路径+新文件名
        os.rename(oldname, newname)  # 改名字
        os.rename(img2npy(oldname), img2npy(newname))  # 改名字
    print('Rename completed')
    pass

def del_file(dir_path,format = None):
    for i in os.listdir(dir_path):  # os.listdir(dir_path)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = dir_path + os.sep + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            if format is None:
                os.remove(file_data)
            else:
                if format in i:
                    os.remove(file_data)
        else:
            del_file(file_data)
    print(dir_path + ' has been cleared')

def clear_dataset(root_dir=r'../dataset'):
    """
    清空dataset文件夹里面的图片和标签
    :param root_dir:
    :return:
    """
    # for root, dirs, files in os.walk(root_dir):
    #     for d in dirs:
    #         print(d)
    #         del_file(os.path.join(root,d))
    del_file(root_dir)

if __name__=='__main__':
    videoPath = '../棋盘视频源文件/06.mp4'
    savePicturePath = '../dataset/Original_dataset'
    clear_dataset()
    video_sampling(videoPath, savePicturePath, interval=2)
    get_undistorted_images(savePicturePath)

    # 暂时不用区分左右了。
    # video_sampling(r'../棋盘视频源文件', savePicturePath)
    # rename_items(savePicturePath)
    # copy_items(savePicturePath,savePicturePath.replace('left','right'))
    # rename_items(savePicturePath)



