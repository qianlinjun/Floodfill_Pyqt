#!/usr/bin/env python
#coding=utf-8
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import numpy as np

from multiprocessing import Process


def skyline_track(img, imgPath):
    # if "rgb" not in imgPath or "crop" in imgPath:
    #     return

    # img = cv2.imread(imgPath)
    [img_h, img_w, img_channel] = img.shape
    trace = []
    start_x = 0
    start_y = 0

    gray = img[:,:,1]

    # print("gray 512 1000", gray[1000, 512], img[512, 512])


    for h in range(img_h):
        for w in range(img_w):
            # mountain 0  blank 255
            if (gray[h,w] > 128):
                gray[h,w] = 0
            else:
                gray[h,w] = 255
    
    #python 跳出多重循环
    #https://www.cnblogs.com/xiaojiayu/p/5195316.html
    class getoutofloop(Exception): pass
    try:
        for w in range(0, img_w - 2):
            for h in range(img_h - 1, -1, -1): 
                if gray[h,w] == 0:
                    start_x = w
                    start_y = h
                    raise getoutofloop
    except getoutofloop:
        pass
    
    
    print("Start Point (%d %d)"%(start_x, start_y))
    trace.append([start_x, start_y])
    
    
    # 8邻域 顺时针方向搜索
    neighbor = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]
    neighbor = list(reversed(neighbor))

    neighbor_len = len(neighbor)
    
    #先从当前点的左上方开始，
    # 如果左上方也是黑点(边界点)：
    #         搜索方向逆时针旋转90 i-=2
    # 否则：
    #         搜索方向顺时针旋转45 i+=1
    i = 0
    cur_x = start_x + neighbor[i][0]
    cur_y = start_y + neighbor[i][1]
    
    is_contour_point = 0
    
    try:
        while not ((cur_x == start_x) and (cur_y == start_y)):
            is_contour_point = 0
            while is_contour_point == 0:
                #neighbor_x = cur_x +
                if gray[cur_y, cur_x] == 0:
                    is_contour_point = 1
                    trace.append([cur_x, cur_y])
                    i -= 2
                    if i < 0:
                        i += neighbor_len
                else:
                    i += 1
                    if i >= neighbor_len:
                        i -= neighbor_len
                #print(i)
                cur_x = cur_x + neighbor[i][0]
                cur_y = cur_y + neighbor[i][1]
    except:
        print("throw error")
    
    # for i in range(len(trace)-1):
    #     cv2.line(img,(trace[i][0],trace[i][1]), (trace[i+1][0], trace[i+1][1]),(0,0,255),3)
    #     cv2.imshow("img", img)
    #     cv2.waitKey(10)
    
    # cv2.rectangle(img,(start_x, start_y),(start_x + 20, start_y + 20),(255,0,0),2)
    # cv2.imshow("img", img)
    # cv2.waitKey(10000)
    # cv2.destroyWindow("img")

    cnt_col, cnt_row = np.mean(trace, 0) #below line cnt
    std_col, std_row = np.std(trace, 0) #below line cnt
    cnt_col, cnt_row = int(cnt_col), int(cnt_row)
    # print(cnt_col, cnt_row, std_col, std_row)
    print("skyline track", cnt_row)

    # render_imgPath = imgPath.replace("_rgb", "")
    # renderImg = cv2.imread(render_imgPath)
    # print(renderImg[cnt_row:, :])
    # cv2.imshow("renderImg", renderImg)

    # renderImg[cnt_row:, :] = [255,255,255]
    # cv2.imshow("cropRenderImg", renderImg)
    # cv2.waitKey(0)
    # cv2.destroyWindow("renderImg")
    # cv2.imwrite(imgPath.replace(".png", "_crop.png"), renderImg)
    return cnt_row



def hist(img):
    # 直方图均衡化
    # (b, g, r) = cv2.split(img)
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)
    # img_output = cv2.merge((bH, gH, rH))
    
    
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output



def simple(img):
    [img_h, img_w, img_channel] = img.shape
    
    gray = img[:,:,1]
    # print("gray 512 1000", gray[1000, 512], gray[512, 512], img[512, 512])

    step = 1
    max_rows = []
    max_rows_pixels = []
    for col_id in range(img_w//step):
        # print("col_id*step : (col_id+1)*step",col_id*step ,(col_id+1)*step)
        mountain_pixels = np.argwhere(gray[:,col_id*step : (col_id+1)*step] > 100)
        if len(mountain_pixels) > 0:
            # min_row, min_col = np.min(mountain_pixels, 0)
            max_row, max_col = np.max(mountain_pixels, 0)
            max_rows.append(max_row)
            max_rows_pixels.append(gray[max(0, max_row-2), col_id*step])
    
    print("simple {} mean pixels{}", np.mean(max_rows), np.mean(max_rows_pixels))
    
    return np.mean(max_rows), np.mean(max_rows_pixels)






def min_max(imgPath):
    if "rgb" not in imgPath or "crop" in imgPath or os.path.isdir(imgPath) is True:
            return
    
    img = cv2.imread(imgPath)
    [img_h, img_w, img_channel] = img.shape

    # print("gray 512 1000", gray[1000, 512], img[512, 512])
    gray = img[:,:,1]
    mountain_pixels = np.argwhere(gray > 128)
    min_row, min_col = np.min(mountain_pixels, 0)
    max_row, max_col = np.max(mountain_pixels, 0)
    
    crop_line_row = int(max_row - (max_row - min_row)/10)


    render_imgPath = imgPath.replace("_rgb", "")
    renderImg = cv2.imread(render_imgPath)
    # print(renderImg[512,512])

    # renderImg_hist =hist(renderImg)
    cv2.imshow("renderImg", renderImg)

    modify_img = fillColor(renderImg, img_h, img_w, crop_line_row)
    
    cv2.imshow("modify_img", modify_img)
    # cv2.imshow("renderImg_hist", renderImg_hist)
    # cv2.imshow("cropRenderImg", renderImg)
    cv2.waitKey(0)
    # cv2.destroyWindow("renderImg")
    # cv2.imwrite(imgPath.replace(".png", "_crop.png"), renderImg)



def transform(imgPath):
    if "rgb" in imgPath or "crop" in imgPath or os.path.isdir(imgPath) is True:
            return

    img = cv2.imread(imgPath)
    cv2.imshow("renderImg", img)
    [img_h, img_w, img_channel] = img.shape

    # gray = img[:,:,1]

    # for h in range(img_h):
    #     for w in range(img_w):
    #         # mountain 0  blank 255
    #         if (gray[h,w] > 230):
    #             img[h,w] = img[h,w] - 20
    #         elif (gray[h,w] < 30):
    #             img[h,w] = img[h,w] + 20
    
    cv2.imshow("renderImg_modify", img)

    # cv2.imshow("cropRenderImg", renderImg)
    # cv2.waitKey(0)
    




def fillColor(img, imgh, imgw, cropRow, mean_pixel=None):

    # if mean_pixel is not None:
    gray = img[:,:,1]
    max_pix = 250
    last_pix = 180
    for w in range(imgw):
        for h in range(cropRow, -1, -1):
            # mountain 0  blank 255
            #停止条件
            if h< cropRow and gray[h,w] < 250:
                
                start_pix = 0.9*last_pix + 0.1*int(gray[h,w]) #和上一个像素也有关系
                last_pix = gray[h,w]#start_pix

                k_slope = (max_pix - start_pix)*1.0/(cropRow - h)
                img[h:cropRow,w] = [ [k_slope*(r - h)+start_pix] * 3 for r in range(h, cropRow)]  #int(mean_pixel)#min(int(gray[h,w]), 255)
                break
            elif h == cropRow:
                last_pix = int(gray[h,w]) #255
    
    
    img[cropRow:, :] = [255,255,255]

    # cv2.imshow("renderImg_modify", img)

    # cv2.imshow("cropRenderImg", renderImg)
    # cv2.waitKey(0)
    return img


def main():
    for imgPath in Path(r"C:\qianlinjun\graduate\test-data\test-simple").iterdir():
        # for imgPath in imgPaths:
        imgPath = str(imgPath)
        if any(["rgb" not in imgPath , "crop" in imgPath , "line" in imgPath, os.path.isdir(imgPath) is True]):
            continue
        
        print(imgPath)
        
        img = cv2.imread(imgPath)
        # img = img[ : , : , (2, 1, 0)]
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        [img_h, img_w, img_channel] = img.shape


        # skyline_row = int(skyline_track(img, imgPath))
        simple_row, mean_pixel =simple(img)
        simple_row, mean_pixel = int(simple_row), int(mean_pixel)

        # cv2.line(img,(0, skyline_row), (1023, skyline_row),(0,0,255),3)
        cv2.line(img,(0, simple_row), (1023, simple_row),(0,0,255),3)
        
        # cv2.imshow("img", img)
        # cv2.waitKey(1000)

    
        # cv2.imwrite(imgPath.replace(".png", "_crop_line.png"), img)

        render_imgPath = imgPath.replace("_rgb", "")
        renderImg = cv2.imread(render_imgPath)
        

        fill_img = fillColor(renderImg, img_h, img_w, simple_row, mean_pixel)
        cv2.imwrite(render_imgPath.replace(".png", "_crop.png"), fill_img)
    
# def muti_process_run(worker_num=10):
#     imgPaths = list(Path(r"C:\qianlinjun\graduate\test-data\test-simple").iterdir())
#     print(imgPaths)
#     workPerProcess = len(imgPaths)//worker_num
#     folders_per_worker = []
#     for i in range(worker_num):
#         _list = imgPaths[i*workPerProcess:(i+1)*workPerProcess ]
#         folders_per_worker.append(_list)
        
#     work_list = []
#     for i in range(worker_num):
#         # print(data, len(data))
#         work_list.append(Process(target=target,name="worker"+str(i), args=( folders_per_worker[i])))
    
#     for work in work_list:
#         work.start()
#     for work in work_list:
#         work.join()








if __name__ == "__main__":
    # testImgPath = r"C:\qianlinjun\graduate\data\switz-test-pts-3-17-11-image-fov-60 -debug\218_rgb_8.57135487_46.6658058.png"
    main()
    # muti_process_run()