import sys
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

colorList = [[255, 0, 51],
            [0, 0, 255],# Blue 
            [0, 255, 0],# Green 
            [0, 255, 255],# Cyan 
            [255, 0, 0],# Red 
            [255, 0, 255],# Magenta 
            [255, 255, 0],# Yellow 
            [255, 255, 255],# White 
            [0, 0, 128],# Dark blue 
            [0, 128, 0],# Dark green 
            [0, 128, 128],# Dark cyan 
            [128, 0, 0],# Dark red 
            [128, 0, 128],# Dark magenta 
            [128, 128, 0],# Dark yellow 
            [128, 128, 128],# Dark gray 
            [192, 192, 192]]# Light gray 

class Polygon(object):
    # id = 0
    def __init__(self, contour, objectId):
        '''
        contour 
        '''
        self.id = objectId
        self.contour = contour
        self.mountain_line = None
        self.fillColor = colorList[self.id % len(colorList)]
    
    def addPoint(self, ptxy):
        if len(np.array(ptxy).shape) == 1:
            ptxy = [[ptxy]]
        elif len(np.array(ptxy).shape) == 2:
            ptxy = [ptxy]
        self.contour = np.concatenate((self.contour, ptxy), axis=0)
    
    def removeLastPoint(self):
        self.contour = self.contour[:-1]

    def getArea(self):
        if len(self.contour) < 3:
            return 0
        else:
            return cv2.contourArea(self.contour)
    
    def getMoments(self):
        # 特征矩
        M = cv2.moments(self.contour)
        # print(M)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # cv2.circle(dst, (cx, cy), 2, (0, 0, 255), -1)   # 绘制圆点
        return [cx,cy]



class PaintArea(QWidget):
    PosChangeSignal = pyqtSignal(int,int)# 定义信号

    def __init__(self):
        super(PaintArea,self).__init__()
        self.left_top_point = None
        self.wait_paint_img     = None
        self.setMinimumSize(800,800)
        

        self.setMouseTracking(True)#get mouse in real time

    def set_pos_and_img(self, left_top_point, img):
        self.left_top_point = left_top_point
        self.wait_paint_img     = img

    def paintEvent(self, e):
        '''
        绘图
        :param e:
        :return:
        '''
        # print("before paintEvent")
        painter = QPainter()#self
        painter.begin(self)
        self.draw_img(painter)
        painter.end()
        # print("after paintEvent")
    
    def draw_img(self, painter):
        if self.left_top_point is None or self.wait_paint_img is None:
            print("draw img args: left_top_point or scale img is none")
            return

        if type(self.wait_paint_img) == QPixmap:
            # print("before drawPixmap")
            painter.drawPixmap(self.left_top_point, self.wait_paint_img)
            # print("after drawPixmap\n")
        else:
            # print("before drawImage")
            painter.drawImage(self.left_top_point, self.wait_paint_img)
            # print("after drawImage\n")
        # self.left_top_point = None
        # self.scaled_img     = None
    
    def mouseMoveEvent(self, e):
        s = e.pos() #e.windowPos()
        x = s.x()
        y = s.y()
        self.PosChangeSignal.emit(x , y)
        


def cvimg_to_qtimg(cvimg):

    height, width, depth = cvimg.shape
    # cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

    return cvimg



def fill_color_demo(img_path, pos=(1,1)):
    image = cv2.imread(img_path)
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray  = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)
    copyIma = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    mask_fill = 252 #mask的填充值
    #floodFill充值标志
    flags = 4|(mask_fill<<8)#|cv2.FLOODFILL_FIXED_RANGE
    
    # mask 需要填充的位置设置为0
    cv2.floodFill(copyIma, mask, pos, 255, 5, 5 , flags)  #(836, 459) , cv2.FLOODFILL_FIXED_RANGE
    # cv.imshow("fill_color", copyIma)
    plt.subplot(1,2,1)
    plt.imshow(image),
    plt.title('image')
    plt.subplot(1,2,2)
    plt.imshow(copyIma),
    plt.title('copyIma')

    plt.show()
    # return mask



def getOnePolygon(mask, ptLocxy):
    """
    img mask
    """
    # img = cv2.imread('002.tif')

    rows, cols = mask.shape
    # 边缘提取
    Ksize = 3
    L2g = True
    # edge = cv2.Canny(mask, 50, 100, apertureSize=Ksize, L2gradient=L2g)

    # 提取轮廓
    '''
    findcontour()函数中有三个参数，第一个img是源图像，第二个model是轮廓检索模式，第三个method是轮廓逼近方法。输出等高线contours和层次结构hierarchy。
    model:  cv2.RETR_EXTERNAL  仅检索极端的外部轮廓。 为所有轮廓设置了层次hierarchy[i][2] = hierarchy[i][3]=-1
            cv2.RETR_LIST  在不建立任何层次关系的情况下检索所有轮廓。
            cv2.RETR_CCOMP  检索所有轮廓并将其组织为两级层次结构。在顶层，组件具有外部边界；在第二层，有孔的边界。如果所连接零部件的孔内还有其他轮廓，则该轮廓仍将放置在顶层。
            cv2.RETR_TREE  检索所有轮廓，并重建嵌套轮廓的完整层次。
            cv2.RETR_FLOODFILL  输入图像也可以是32位的整型图像(CV_32SC1)
    method：cv2.CHAIN_APPROX_NONE  存储所有的轮廓点，任何一个包含一两个点的子序列（不改变顺序索引的连续的）相邻。
            cv2.CHAIN_APPROX_SIMPLE  压缩水平，垂直和对角线段，仅保留其端点。 例如，一个直立的矩形轮廓编码有4个点。
            cv2.CHAIN_APPROX_TC89_L1 和 cv2.CHAIN_APPROX_TC89_KCOS 近似算法
    '''

    mask[0:2, :] = 0
    mask[:  ,0:2] = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # if len(contours) > 0:
    #     dst = np.ones(mask.shape, dtype=np.uint8)
    #     # 绘制轮廓 第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
    #     cv2.drawContours(dst, contours, -1, (255, 0, 0), cv2.FILLED)
    #     cv2.imshow("dst", dst)
    #     cv2.waitKey()

    print("find contours", len(contours))

    if len(contours) == 0:
       return False, None

    min_area = 10000000000000000000000000
    min_idx = -1
    #find min area
    for idx, cnt in enumerate(contours):
        
        area = cv2.contourArea(cnt)
        ifInPoly = cv2.pointPolygonTest(cnt, ptLocxy ,True)
        # print("ifinpoly", ifInPoly, " pts num", len(cnt))
        # # ifInPoly is True and
        # print("area", area)
        # and area < 1440000*0.7
        # 
        if  ifInPoly > 0 and (area > 15 and area < min_area):
            min_idx = idx
            min_area = area

    if min_idx < 0:
       return False, None

    cnt = contours[min_idx]

    # print("min_area {} \n".format(min_area) )
    # dst = np.ones(mask.shape, dtype=np.uint8)
    # cv2.drawContours(dst, [cnt], -1, (255, 0, 0), cv2.FILLED)#1)
    # cv2.imshow("dst", dst)
    # cv2.waitKey()

    # # 轮廓面积
    # area = cv2.contourArea(cnt)
    # print("minarea", area)

    # # 特征矩
    # M = cv2.moments(cnt)
    # # print(M)
    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])
    # cv2.circle(dst, (cx, cy), 2, (0, 0, 255), -1)   # 绘制圆点



    # 轮廓周长：第二个参数指定形状是闭合轮廓(True)还是曲线
    # perimeter = cv2.arcLength(cnt, True)
    # print(perimeter)

    # 轮廓近似：epsilon是从轮廓到近似轮廓的最大距离--精度参数
    # epsilon = 0.01 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # cv2.polylines(dst, [approx], True, (0, 255, 255))   # 绘制多边形
    # print(approx)

    # 轮廓凸包：returnPoints：默认情况下为True。然后返回凸包的坐标。如果为False，则返回与凸包点相对应的轮廓点的索引。
    # hull = cv2.convexHull(cnt, returnPoints=True)
    # cv2.polylines(dst, [hull], True, (255, 255, 255), 2)   # 绘制多边形
    # print(hull)

    # 检查凸度：检查曲线是否凸出的功能，返回True还是False。
    # k = cv2.isContourConvex(cnt)
    # print(k)

    # 直角矩形
    # x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(dst, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    # 旋转矩形
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(dst, [box], 0, (0, 0, 255), 2)

    # 最小外接圆
    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    # center = (int(x), int(y))
    # radius = int(radius)
    # cv2.circle(dst, center, radius, (0, 255, 0), 2)

    # 拟合椭圆
    # ellipse = cv2.fitEllipse(cnt)
    # cv2.ellipse(dst, ellipse, (0, 0, 255), 2)

    # 拟合直线
    # rows, cols = mask.shape[:2]
    # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # lefty = int((-x*vy/vx) + y)
    # righty = int(((cols-x)*vy/vx)+y)
    # cv2.line(dst, (cols-1, righty), (0, lefty), (255, 255, 255), 2)

    return True, cnt #Polygon(cnt)



# 求两个ROI的交并补等操作
def mergePoly(ori_hw, poly1, poly2):
    #1， 创建第一个image与mask
    img_zero = np.zeros((ori_hw[0], ori_hw[1], 3), dtype=np.uint8)
    img_zero_sample = img_zero.copy()
    img_zero_2 = img_zero.copy()
    print(img_zero.shape)

    # points = np.array([[101, 101], [110, 101], [101, 110], [110, 110]])
    # points_ROI = np.array([[101, 101], [110, 101], [110, 110], [101, 110]])
    # points_ROI = np.array([[101, 101], [300, 101], [300, 300], [101, 300]])
    # 画顶点
    # for i in range(len(points)):
    #     cv2.circle(img_zero, tuple(points[i]), 5, (0,255,255), -1)

    # 填充mask
    # cv2.fillPoly(img_zero, [points], (0, 0, 255))  # 任意多边形
    # cv2.fillPoly(img_zero, poly1, (255, 255, 255))  # 凸包
    cv2.drawContours(img_zero, [poly1], -1, (255, 255, 255), cv2.FILLED)#1)
    poly1_gray = cv2.cvtColor(img_zero, cv2.COLOR_RGB2GRAY)

    # img_gray_roi_mask中mask的面积，即为白色的点
    mask = poly1_gray[poly1_gray > 0]
    print(len(mask)) # 100个点

    cv2.imshow('poly1_gray A', poly1_gray)

    #2, 创建第二个image与mask
    # points_ROI_triangle = np.array([[201, 100], [400, 100], [300, 300]])
    # 填充maskfillConvexPoly
    # cv2.fillPoly(img_zero_2, poly2, (0, 255, 255))  # 凸包
    cv2.drawContours(img_zero_2, [poly2], -1, (255, 255, 255), cv2.FILLED)
    poly2_gray = cv2.cvtColor(img_zero_2, cv2.COLOR_RGB2GRAY)
    #灰度化后的图转化为白色255
    poly2_gray[poly2_gray > 0] = 255
    # cv2.imshow('poly2_gray B', poly2_gray)

    #3.1， 求 img_gray_roi_mask(named: A) 与 img_gray_roi_mask_triangle(named: B) 的并集
    A_or_B = cv2.bitwise_or(poly1_gray, poly2_gray)
    # cv2.imshow('A_or_B', A_or_B)

    #3.2， 求 img_gray_roi_mask(named: A) 与 img_gray_roi_mask_triangle(named: B) 的交集
    A_and_B = cv2.bitwise_and(poly1_gray, poly2_gray)
    # cv2.imshow('A_and_B', A_and_B)

    print(len(A_and_B[A_and_B > 0] ))

    # 3.3， 求属于A，但不属于B的部分：A-AB
    # A_sub_AB = cv2.bitwise_xor(img_gray_roi_mask, A_and_B)
    # cv2.imshow('A_sub_AB', A_sub_AB)

    # 3.4， 求A与B的并集关于整张图片的补集，即求整张图片扣去A和B
    # A_or_B_not = cv2.bitwise_not(A_or_B)
    # cv2.imshow('A_or_B_not', A_or_B_not)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def savePolygons2Json(img_name, polygons_list):
    if "png" not in img_name:
        print("invalid for image:{}".format(img_name))
        return

    save_file_path = os.path.join("./", img_name[:-4] + ".json") 
    with open(save_file_path) as save_f:
        list_2_save = []
        for poly in polygons_list:
            if len(poly.contour) >= 3:
                cur_poly = {}
                # self.id = objectId
                # self.contour = contour
                # self.mountain_line = None
                # self.fillColor
                cur_poly["id"]        = poly.id
                cur_poly["contour"]   = poly.contour
                cur_poly["fillColor"] = poly.fillColor
                cur_poly["area"]      = poly.getArea()
                cur_poly["momente"]   = poly.getMoments()
                list_2_save.append(cur_poly)
        if len(list_2_save) > 0:
            data_2_save = json.dumps(list_2_save, ensure_ascii=False)
            save_f.write(data_2_save.encode('utf-8'))




class Operation(object):
    '''
    1. insert a polygon by floodfill
    2. draw a polygon by hand, by draw points then by then
    3. merge two poygons
    '''
    insertPolygonType = "insertPolygon"
    drawPolygonByHandType = "drawPolygonByHand"
    mergePolygonsType = "mergePolygons"
    def __init__(self, operateType, relateObjectIds, relatePosxy = None):
        assert operateType in [Operation.insertPolygonType, Operation.drawPolygonByHandType, Operation.mergePolygonsType]
        self.operateType     = operateType
        self.relateObjectIds = relateObjectIds
        self.relatePos       = relatePosxy
    




# if __name__ == "__main__":
#   fill_color_demo(sys.argv[1], (811,490))


# fill flood
# img1 = cv2.imread(sys.argv[1])
# gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# gray  = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)


# plt.imshow(gray),
# plt.title('Original')
# plt.show()

# # Otsu ' s 二值化；
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)#_INV
# print("ret", ret)

# plt.subplot(1,3,1),
# plt.imshow(gray2),
# plt.title('Original')
# plt.subplot(1,3,3),
# plt.imshow(thresh),
# plt.title("thresh")
# plt.show()

# #nosing removoal迭代两次
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
 
# # sure background area
# sure_bg = cv2.dilate(opening, kernel,iterations = 3)
# plt.imshow(opening),
# plt.title('opening')
# plt.imshow(sure_bg),
# plt.title('sure_bg')
# plt.show()

# dist_transform = cv2.distanceTransform(opening,1,5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)
 
# sure_fg = np.uint8(sure_fg)
# unknow = cv2.subtract(sure_bg,sure_fg)
 
 
# plt.subplot(1,3,1),
# plt.imshow(sure_bg),
# plt.title('Black region\n must be background')
 
# plt.subplot(1,3,3),
# plt.imshow(unknow),
# plt.title('Yellos region\n must be foregroun')


# #Marker labeling
# ret, makers1 = cv2.connectedComponents(sure_fg)
 
# #Add one to all labels so that sure background is not 0 but 1;
# markers = makers1 +1
 
# #Now mark the region of unknow with zero;
# markers[unknow ==255] =0
# markers3 = cv2.watershed(img1, markers)


# img1[markers3 == -1] =[255,0,0]
 
# plt.subplot(1,3,1),
# plt.imshow(makers1),
# plt.title('makers1')
# plt.subplot(1,3,2),
# plt.imshow(markers3),
# plt.title('markers3')
# plt.subplot(1,3,3),
# plt.imshow(img1),
# plt.title('img1')

# plt.show()

