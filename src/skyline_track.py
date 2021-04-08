#!/usr/bin/env python
#coding=utf-8
 
import cv2
 

def skyline_track():
    img = cv2.imread(r"C:\qianlinjun\graduate\data\switz-test-pts-3-17-11-image-fov-60 -debug\1_rgb_8.63674355_46.8405838.png")
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
    
    for i in range(len(trace)-1):
        cv2.line(img,(trace[i][0],trace[i][1]), (trace[i+1][0], trace[i+1][1]),(0,0,255),3)
        cv2.imshow("img", img)
        cv2.waitKey(10)
    
    cv2.rectangle(img,(start_x, start_y),(start_x + 20, start_y + 20),(255,0,0),2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyWindow("img")



if __name == "__main__":
