import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles=None, num_cols=None, scale=3, normalize=False):
    """ 一个窗口中绘制多张图像:
    Args: 
        images: 可以为一张图像(不要放在列表中)，也可以为一个图像列表
        titles: 图像对应标题、
        num_cols: 每行最多显示多少张图像
        scale: 用于调整图窗大小
        normalize: 显示灰度图时是否进行灰度归一化
    """

    # 加了下面2行后可以显示中文标题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 单张图片显示
    if not isinstance(images, list):
        if not isinstance(scale, tuple):
            scale = (scale, scale * 1.5)

        plt.figure(figsize=(scale[1], scale[0]))
        img = images
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为BGR通道，需要转换一下
            B, G, R = cv.split(img)
            img = cv.merge([R, G, B])
            plt.imshow(img)
        elif len(img.shape) == 2:
            # pyplot显示灰度需要加一个参数
            if normalize:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                            str(img.shape) + " of image data")
        if titles is not None:
            plt.title(titles, y=-0.15)
        plt.axis('off')
        plt.show()
        return

    # 多张图片显示
    if not isinstance(scale, tuple):
        scale = (scale, scale)

    num_imgs = len(images)
    if num_cols is None:
        num_cols = int(np.ceil((np.sqrt(num_imgs))))
    num_rows = (num_imgs - 1) // num_cols + 1

    idx = list(range(num_imgs))
    _, figs = plt.subplots(num_rows, num_cols,
                           figsize=(scale[1] * num_cols, scale[0] * num_rows))
    for f, i, img in zip(figs.flat, idx, images):
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为BGR通道，需要转换一下
            B, G, R = cv.split(img)
            img = cv.merge([R, G, B])
            f.imshow(img)
        elif len(img.shape) == 2:
            # pyplot显示灰度需要加一个参数
            if normalize:
                f.imshow(img, cmap='gray')
            else:
                f.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                            str(img.shape) + " of image data")
        if titles is not None:
            f.set_title(titles[i], y=-0.15)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # 将不显示图像的fig移除，不然会显示多余的窗口
    if len(figs.shape) == 1:
        figs = figs.reshape(-1, figs.shape[0])
    for i in range(num_rows * num_cols - num_imgs):
        figs[num_rows - 1, num_imgs % num_cols + i].remove()
    plt.show()

if __name__ ==  "__main__":
    new_img_paths = []
    for img_path in glob.glob(r"D:\qianlinjun\graduate\src\src\output\4_14_with_edge_cost_no_angle_same_left_right\10_8.589164277703366_46.63544153957453\vis\*.png"):
        # print(img_path)
        # print(int(img_path.split("\\")[-1].split("_")[0]))
        new_img_paths.append([img_path, int(img_path.split("\\")[-1].split("_")[0])])
    
    order_imgpaths = sorted(new_img_paths, key=lambda item: item[1])
    fin_paths = []
    for path in order_imgpaths:
        img = cv.imread(path[0])
        fin_paths.append(img)
    titles = ["1 编辑距离:653.95","2 编辑距离:739.58","3 编辑距离:782.60","4 编辑距离:815.16",\
    "5 编辑距离:867.64","6 编辑距离:867.64 ","7 编辑距离:903.12","8 编辑距离:909.46",\
    "9 编辑距离:941.15","10 编辑距离:968.14","11 编辑距离:1058.86","12 编辑距离:1069.65"]
    show_images(fin_paths,titles,num_cols=3)
