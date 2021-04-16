import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
# 设置matplotlib显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei'] # 显示中文为黑体
matplotlib.rcParams['axes.unicode_minus']=False # 正常显示负号


def muti_pic():
    # no humoments 
    plt.subplot(2,2,1)
    x1 = np.arange(25) 
    y1 = [6, 7, 9, 10, 14, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 20]
    plt.title("直线图") 
    plt.xlabel("x轴") 
    plt.ylabel("y轴") 
    plt.plot(x1,y1,'-.ro') 


    plt.subplot(2,2,2)
    x2 = np.arange(25) 
    y2 =  x2 / 2
    plt.plot(x2,y2,':b+') 

    plt.subplot(2,2,1)
    x1 = np.arange(25) 
    y1 =  2 * x1
    plt.title("直线图") 
    plt.xlabel("x轴") 
    plt.ylabel("y轴") 
    plt.plot(x1,y1,'-.ro') 

    plt.subplot(2,2,1)
    x1 = np.arange(25) 
    y1 =  2 * x1
    plt.title("直线图") 
    plt.xlabel("x轴") 
    plt.ylabel("y轴") 
    plt.plot(x1,y1,'-.ro') 


    plt.show()


def node_exper():
    x1 = np.arange(25) 
    # no edge cost
    # y1 =  np.array([6, 8, 11, 12, 12, 12, 13, 13, 15, 15, 15, 15, 16, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 20, 20])/25.
    # no angle dist
    y1 = np.array([6, 7, 9, 10, 14, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 20])/25. #原始算法
    y2 =  np.array([4, 6, 10, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])/25. # no iou factor
    y3 = np.array([5, 6, 9, 10, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 16, 16, 16, 16, 16, 16])/25. #no angle factor
    # y4 = np.array([7, 7, 8, 10, 10, 11, 12, 15, 16, 16, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18])/25. #no area factor
    y4 = np.array([6, 9, 11, 12, 12, 12, 13, 13, 15, 15, 15, 15, 16, 16, 17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 21])/25.
    # plt.title("直线图") 
    plt.xlabel("候选结果数量N") 
    plt.ylabel("图像正确定位比例") 
    plt.ylim(0, 1)
    plt.plot(x1,y1,'b',label="本文方法")#'-.ro' 
    plt.plot(x1,y2,'--r' ,label="去掉交并比因子")#
    plt.plot(x1,y3,'-.g',label="去掉角度因子")#'-.ro' 
    plt.plot(x1,y4, color='y',linestyle=":",label="去掉面积因子")#
    plt.legend(loc="100,100")
    plt.show()

def edge_exper():
    x1 = np.arange(25) 
    
    # ori no angle dist
    y1 = np.array([6, 7, 9, 10, 14, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 20])/25. #原始算法
    # no edge cost
    y2 =  np.array([6, 8, 11, 12, 12, 12, 13, 13, 15, 15, 15, 15, 16, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 20, 20])/25.
    y3 =  np.array([5, 8, 10, 10, 13, 14, 14, 15, 16, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19])/25.
    # y2 =  np.array([4, 6, 10, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])/25. # no iou factor
    # y3 = np.array([5, 6, 9, 10, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 16, 16, 16, 16, 16, 16])/25. #no angle factor
    # y4 = np.array([7, 7, 8, 10, 10, 11, 12, 15, 16, 16, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18])/25. #no area factor
    # plt.title("直线图") 
    plt.xlabel("候选结果数量N") 
    plt.ylabel("图像正确定位比例") 
    plt.ylim(0, 1)
    plt.plot(x1,y1,'b',label="本文方法")#'-.ro' 
    plt.plot(x1,y2,'--r' ,label="去掉边距离代价")#
    plt.plot(x1,y3,'-.g',label="加入边角度代价")#'-.ro' 
    # plt.plot(x1,y4, color='y',linestyle=":",label="去掉面积因子")#
    plt.legend(loc="0,0")
    plt.show()


def cmp():
    x1 = np.arange(25) 
    
    # ori no angle dist
    y1 = np.array([6, 7, 9, 10, 14, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 20])/25. #原始算法

    y2 =  np.array([7, 9, 11, 13, 15, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 20, 20, 20, 20, 20])/25. #1.1
    y3 = np.array([6, 8, 10, 14, 14, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20])/25. #1_0
    y4 = np.array([7, 10, 11, 13, 15, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20])/25. #2_0
    y5 = np.array([7, 10, 11, 13, 15, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20])/25.
  
    # plt.title("直线图") 
    plt.xlabel("候选结果数量N") 
    plt.ylabel("图像正确定位比例") 
    plt.ylim(0, 1)
    plt.plot(x1,y1,'b',label="本文方法")#'-.ro' 
    plt.plot(x1,y2,'--r' ,label="1_1")#
    plt.plot(x1,y3,'-.g',label="1_0")#'-.ro' 
    plt.plot(x1,y4, color='y',linestyle=":",label="2_0")#
    plt.plot(x1,y5,'.c',label="1_5")#'-.ro' 
    plt.legend(loc="0,0")
    plt.show()

# node_exper()
# edge_exper()

cmp()