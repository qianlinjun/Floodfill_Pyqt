import os
import sys
import cv2
import json
import math
import numpy as np
from shapely.geometry import Polygon, Point

# with open('out.txt', 'w+') as file:

fsock = open('C:\qianlinjun\graduate\gen_dem\output\src\out.txt', 'a+')
sys.stdout = fsock
# print 'This message is for file!'
print("\n\n-----------------------------------------\n")

def show_polygon(poly):
    dst = np.ones((1500, 1500), dtype=np.uint8)
    perimeter = cv2.arcLength(poly, True)
    # print(perimeter)

    # 轮廓近似：epsilon是从轮廓到近似轮廓的最大距离--精度参数
    epsilon = 0.01 * cv2.arcLength(poly, True)
    # approx = cv2.approxPolyDP(poly2, epsilon, True)
    cv2.polylines(dst, [poly], True, (255, 255, 255))   # 绘制多边形


    # x, y, w, h = cv2.boundingRect(poly2)
    # cv2.rectangle(dst, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('dst A', dst)

    cv2.waitKey(0)


def get_iou(poly1, poly2):
    '''
    将两个多边形根据质心对齐
    poly1: 多边形
    poly2: 多边形

    '''
    m1 = cv2.moments(poly1)
    cx1, cy1=m1['m10'] / m1['m00'], m1['m01'] / m1['m00']
    # print(cx1, cy1)

    m2 = cv2.moments(poly2)
    cx2, cy2=m2['m10'] / m2['m00'], m2['m01'] / m2['m00']
    # print(cx2, cy2)

    #将poly2 移到 poly2
    polygon1 = Polygon(poly1)#.convex_hull
    poly1_area = polygon1.area
    

    # polygon1 = polygon1.buffer(0) # 加一个较小的buffer
    # print("old poly1", poly1, polygon1.is_simple)
    # print("old poly2", poly2, Polygon(poly2).area)
    poly2[:,0] = poly2[:,0] - (cx2 - cx1)
    poly2[:,1] = poly2[:,1] - (cy2 - cy1)
    # print("new poly2", poly2)

    


    polygon2 = Polygon(poly2)#.buffer(0)#.convex_hull
    poly2_area = polygon2.area

    # 暴力计算
    # small_polyarray, big_polyarray = (poly1, poly2) if  poly2_area > poly1_area else (poly2, poly1)
    # small_polygon, big_polygon = (polygon1, polygon2) if  poly2_area > poly1_area else (polygon2, polygon1)
    # # print("small_area:{}".format(small_polygon.area))
    # x, y, w, h = cv2.boundingRect(small_polyarray)
    # # small_area = 0
    # inter_area = 0
    # itersec_area = 0
    # for px in range(x, x+w+1):
    #     for py in range(y, y+h+1):
    #         if small_polygon.contains(Point(px, py)):
    #             small_area += 1
    #             if big_polygon.contains(Point(px, py)):
    #                 inter_area += 1

    #         # if small_polygon.intersects(Point(px, py)):
    #         #     if big_polygon.intersects(Point(px, py)):
    #         #         itersec_area += 1
    # # print("recal small_area:{}".format(small_area))

    # # print("poly1 area poly2 area inter area", poly1_area, polygon2.area, inter_area, itersec_area)
    

    # 调用shapely intersection 接口计算
    # print("new poly2", polygon2.area )
    # polygon2 = polygon2.buffer(0.999999999999)
    # print("new poly2 buff", polygon2.boundary.xy )
    # print("poly1 isvalid::{} poly2 isvalid:{}".format(polygon1.is_valid, polygon1.is_valid))
    # inter_area = polygon1.buffer(0.01).intersection(polygon2.buffer(0.01)).area  #相交面积
    # # inter_area = (inter_area + itersec_area
    

    poly1_area , poly2_area, inter_area = bitwise_and_iou(poly1, poly2)
    union_area = poly1_area + poly2_area - inter_area
    print("poly1 area {}  poly2 area {} inter area {}".format(poly1_area , poly2_area , inter_area))



    
    # exit(0)

    return poly1_area , poly2_area, inter_area*1.0/union_area



def bitwise_and_iou(poly1, poly2):
    dst1 = np.ones([1200,1200], dtype=np.uint8)
    cv2.drawContours(dst1, [poly1], -1, (255, 0, 0), cv2.FILLED)#1)
    

    dst2 = np.ones([1200,1200], dtype=np.uint8)
    cv2.drawContours(dst2, [poly2], -1, (255,0, 0), cv2.FILLED)#1)
    
    img1_bg = cv2.bitwise_and(dst1, dst2)
    # print(np.max(img1_bg), np.min(img1_bg), np.mean(img1_bg))
    
    dst1_area  = np.sum(dst1==255) 
    dst2_area  = np.sum(dst2==255) 
    merge_area = np.sum(img1_bg==255) 
    
    # cv2.imshow("dst1:{}".format(np.sum(dst1==255)), dst1)
    # cv2.imshow("dst2:{}".format(np.sum(dst2==255)), dst2)
    # cv2.imshow("img1_bg:{}".format(np.sum(img1_bg==255)), img1_bg)
    # cv2.waitKey()
    print("pixel_and_iou poly1:{} poly2:{} merge:{}".format(dst1_area, dst2_area, merge_area))
    return dst1_area, dst2_area, merge_area

# if polygon.contains(point):
#         print ('Found containing polygon:', feature)


def hu_moments(poly1, poly2):
    '''
    找到两个形状之间的相似性
    '''
    # 函数返回值代表相似度大小，完全相同的图像返回值是0，返回值最大是1
    # https://www.cnblogs.com/farewell-farewell/p/6004313.html第三个参数
    value = cv.matchShapes(poly1, poly2,  0, 0.0)
	# if value < min_value:
	# 	min_value = value
	# 	min_pos = i

if __name__ ==  "__main__":
    # poly1 = [[284, 485], [283, 486], [274, 486], [273, 487], [264, 487], [263, 488], [263, 489], [265, 489], [266, 490], [275, 490], [276, 491], [283, 491], [284, 492], [290, 492], [291, 493], [295, 493], [296, 494], [300, 494], [301, 495], [304, 495], [305, 496], [308, 496], [309, 497], [313, 497], [314, 498], [317, 498], [318, 499], [321, 499], [322, 500], [325, 500], [326, 501], [329, 501], [330, 502], [333, 502], [334, 503], [337, 503], [338, 504], [341, 504], [342, 505], [345, 505], [346, 506], [348, 506], [349, 507], [352, 507], [353, 508], [355, 508], [356, 509], [359, 509], [360, 510], [367, 510], [368, 509], [369, 509], [370, 508], [371, 508], [372, 507], [373, 507], [374, 506], [375, 506], [377, 504], [378, 504], [379, 503], [380, 503], [382, 501], [383, 501], [385, 499], [386, 499], [387, 498], [388, 498], [389, 497], [389, 496], [374, 496], [373, 495], [370, 495], [369, 494], [366, 494], [365, 493], [362, 493], [361, 492], [358, 492], [357, 491], [353, 491], [352, 490], [348, 490], [347, 489], [338, 489], [337, 490], [329, 490], [328, 491], [327, 490], [318, 490], [317, 489], [312, 489], [311, 488], [308, 488], [307, 487], [305, 487], [304, 486], [301, 486], [300, 485]]
    # poly2 = [[544, 447], [543, 448], [517, 448], [516, 449], [513, 449], [512, 450], [508, 450], [507, 451], [504, 451], [503, 452], [499, 452], [498, 453], [494, 453], [493, 454], [491, 454], [490, 455], [489, 455], [488, 456], [487, 456], [486, 457], [485, 457], [484, 458], [483, 458], [482, 459], [481, 459], [479, 461], [478, 461], [477, 462], [476, 462], [475, 463], [474, 463], [473, 464], [472, 464], [471, 465], [470, 465], [468, 467], [467, 467], [466, 468], [465, 468], [464, 469], [463, 469], [462, 470], [461, 470], [460, 471], [459, 471], [458, 472], [456, 472], [455, 473], [454, 473], [453, 474], [452, 474], [451, 475], [450, 475], [449, 476], [448, 476], [447, 477], [445, 477], [444, 478], [443, 478], [442, 479], [441, 479], [440, 480], [439, 480], [438, 481], [436, 481], [435, 482], [434, 482], [433, 483], [432, 483], [431, 484], [430, 484], [429, 485], [428, 485], [427, 486], [425, 486], [424, 487], [421, 487], [420, 488], [416, 488], [415, 489], [412, 489], [411, 490], [408, 490], [407, 491], [403, 491], [402, 492], [399, 492], [398, 493], [395, 493], [394, 494], [392, 494], [390, 496], [389, 496], [387, 498], [386, 498], [385, 499], [384, 499], [382, 501], [381, 501], [379, 503], [378, 503], [377, 504], [376, 504], [374, 506], [373, 506], [371, 508], [370, 508], [369, 509], [368, 509], [368, 510], [389, 510], [390, 511], [405, 511], [406, 510], [415, 510], [416, 509], [420, 509], [421, 508], [425, 508], [426, 507], [430, 507], [431, 506], [435, 506], [436, 505], [437, 505], [438, 504], [440, 504], [441, 503], [442, 503], [443, 502], [444, 502], [445, 501], [446, 501], [447, 500], [448, 500], [449, 499], [450, 499], [451, 498], [452, 498], [453, 497], [454, 497], [456, 495], [457, 495], [458, 494], [459, 494], [460, 493], [461, 493], [462, 492], [463, 492], [464, 491], [465, 491], [466, 490], [467, 490], [468, 489], [469, 489], [470, 488], [471, 488], [472, 487], [473, 487], [474, 486], [475, 486], [477, 484], [478, 484], [479, 483], [480, 483], [481, 482], [483, 482], [484, 481], [485, 481], [486, 480], [488, 480], [489, 479], [490, 479], [491, 478], [492, 478], [493, 477], [495, 477], [497, 475], [499, 475], [500, 474], [501, 474], [502, 473], [504, 473], [505, 472], [506, 472], [507, 471], [509, 471], [510, 470], [511, 470], [512, 469], [514, 469], [515, 468], [516, 468], [517, 467], [519, 467], [520, 466], [521, 466], [522, 465], [523, 465], [524, 464], [526, 464], [528, 462], [529, 462], [530, 461], [531, 461], [532, 460], [533, 460], [534, 459], [535, 459], [536, 458], [537, 458], [538, 457], [539, 457], [540, 456], [541, 456], [542, 455], [543, 455], [544, 454], [545, 454], [546, 453], [547, 453], [548, 452], [549, 452], [551, 450], [552, 450], [553, 449], [554, 449], [555, 448], [556, 448], [556, 447]]
    filedir  = "C:\qianlinjun\graduate\gen_dem\output\img_with_mask\switz-100-points"

    # 85_8.58807087_46.664257
    json_file1 = os.path.join(filedir, "83_8.58592987_46.6606636.json")#"10_8.51592159_46.602951.json"
    json_file2 = os.path.join(filedir, "85_8.58807087_46.664257.json")#"11_8.53155708_46.60886.json"

    # 2,4,
    # 
    for point1_id in [2,4,5,8]:
        for point2_id in [2,3,4,6,10]:
            print(point1_id, point2_id)
            # point1_id = 8#2#8#2#1#8#0#4#3
            # point2_id = 2#1#11#5#5#2#6#5#0

            poly_json1 = json.load(open(json_file1,'r'))
            poly_json2 = json.load(open(json_file2,'r'))

            poly1, poly2 = None, None
            for idx, polygon in enumerate(poly_json1):
                id_     = polygon["id"]
                if id_ == point1_id:
                    poly1 = polygon["contour"]
                    for idx, polygon in enumerate(poly_json2):
                        id_     = polygon["id"]
                        if id_ == point2_id:
                            poly2 = polygon["contour"]
                            
            if poly1 is not None and poly2 is not None:
                poly1 = np.array(poly1).squeeze()
                poly2 = np.array(poly2).squeeze()

                # # if point1_id == 4:
                # #     show_polygon(poly1)
                poly1_area, poly2_area, iou = get_iou(poly1, poly2)
                # poly1_area, poly2_area, iou = bitwise_and_iou(poly1, poly2)
                # print("iou", iou)
                # # exit(0)
                
                # hu = cv2.HuMoments( m ) hu 表示返回的Hu 矩阵，参数m 是cv2.moments() 
                m1 = cv2.moments(poly1) # m原始 mu 中心化 nu 归一化
                hu1 = cv2.HuMoments( m1 )
                # print(hu1)

                # print(m1["nu02"], m1["nu20"])
                # m = cv2.moments(poly2) # m原始 mu 中心化 nu 归一化
                # print(m["nu02"], m["nu20"])
                mu11 = m1["nu11"]
                mu02 = m1["nu02"]
                mu20 = m1["nu20"]
                angle1 = math.atan(2*mu11/(mu20 - mu02))/2.*180/3.1415926
                # print("poly1 angle", angle1) #*180/3.1415926


                m2 = cv2.moments(poly2)
                hu2 = cv2.HuMoments( m2 )
                mu11 = m2["nu11"]
                mu02 = m2["nu02"]
                mu20 = m2["nu20"]
                angle2 = math.atan(2*mu11/(mu20 - mu02))/2.*180/3.1415926
                # print("poly2 angle", angle2) #*180/3.1415926

                # print(hu2)
                dis = 0
                hu_feature = 0

                # if( ama > eps && amb > eps )
                #     {
                #         ama = sma * log10( ama );
                #         amb = smb * log10( amb );
                #         result += fabs( -ama + amb );
                #     }
                eps = 1e-5
                idx = 0
                for hu1_m, hu2_m in zip(hu1, hu2):
                    # if idx == 4:
                    #     break
                    idx += 1
                    # print(hu1_m, hu2_m)
                    if abs(hu1_m) > eps and  abs(hu2_m) > eps:
                        # angle1 *  angle2 *
                        dis += abs(  np.sign(hu2_m)*math.log10(abs(hu2_m)) -  np.sign(hu1_m)*math.log10(abs(hu1_m)) ) * 1. / (iou + 0.1) * max(poly1_area, poly2_area) / min(poly1_area, poly2_area)
                        # hu_feature += abs(  np.sign(hu2_m)*math.log10(abs(hu2_m)) -  np.sign(hu1_m)*math.log10(abs(hu1_m)) )  ##-np.sign(hu2_m)*math.log10(abs(hu2_m))
                        # dis += abs(  hu2_m -  hu1_m )
                

                print("{} {} matchShapes {}".format( point1_id, point2_id, cv2.matchShapes(poly1, poly2,  2, 0.0))) # 相似度越大 matchshape值越小 iou 越靠近1
                print("{} {} hand dis {} ".format( point1_id, point2_id, dis[0]))
                # print("hu_feature", hu_feature)
                print("\n")
            
            elif poly1 is None:
                print("poly1 is None")
            elif poly2 is None:
                print("poly2 is None")



fsock.close()



# 普通矩：
# 0阶矩（m00）:目标区域的质量 
# 1阶矩（m01,m10）：目标区域的质心 
# 2阶矩（m02,m11,m20）：目标区域的旋转半径 
# 3阶矩（m03,m12,m21,m30）：目标区域的方位和斜度，反应目标的扭曲

# 质心对准之后计算iou




# poly1 area 192851.5  poly2 area 9263.0 inter area 8932
# abs_area: 609.036538979171
# poly1 area 192851.5  poly2 area 29814.0 inter area 29023
# abs_area: 165.58721299326365
# poly1 area 192851.5  poly2 area 190454.0 inter area 179976
# abs_area: 0.4316877586858847
# poly1 area 45436.5  poly2 area 9263.0 inter area 8299
# abs_area: 314.0553808145101
# poly1 area 45436.5  poly2 area 29791.0 inter area 22474
# abs_area: 42.19098974967348
# poly1 area 45436.5  poly2 area 190209.0 inter area 42435
# abs_area: 219.62137909637116
# poly1 area 7499.5  poly2 area 9263.0 inter area 5388
# abs_area: 25.211741866752444
# poly1 area 7499.5  poly2 area 29814.0 inter area 6270
# abs_area: 141.19602862436847
# poly1 area 7499.5  poly2 area 190454.0 inter area 7117
# abs_area: 2190.644763139795
# dist:187.888111067158
# poly1 area 192851.5  poly2 area 2020.0 inter area 1851
# abs_area: 785.5662817229307
# poly1 area 192851.5  poly2 area 7067.0 inter area 6844
# abs_area: 315.86462770086416
# poly1 area 192851.5  poly2 area 94787.5 inter area 78470
# abs_area: 5.727135003642804
# poly1 area 45436.5  poly2 area 2020.0 inter area 1851
# abs_area: 1720.6418436587892
# poly1 area 45436.5  poly2 area 7067.0 inter area 6620
# abs_area: 420.38249561048474
# poly1 area 45436.5  poly2 area 94787.5 inter area 18390
# abs_area: 148.36831083030327
# poly1 area 7499.5  poly2 area 2020.0 inter area 1461
# abs_area: 143.59790269786643
# poly1 area 7499.5  poly2 area 7067.0 inter area 2423
# abs_area: 45.839433079656814
# poly1 area 7499.5  poly2 area 94801.5 inter area 6963
# abs_area: 958.4961429127113
# dist:854.6357893890478
# poly1 area 192851.5  poly2 area 64160.5 inter area 61133
# abs_area: 31.961438634043915
# poly1 area 192851.5  poly2 area 166310.5 inter area 160669
# abs_area: 1.4410428616280502
# poly1 area 45436.5  poly2 area 64185.0 inter area 35476
# abs_area: 27.964040030285034
# poly1 area 45436.5  poly2 area 166298.5 inter area 42244
# abs_area: 165.47685997019144
# poly1 area 7499.5  poly2 area 64244.5 inter area 7117
# abs_area: 423.6975747144786
# poly1 area 7499.5  poly2 area 166497.5 inter area 7117
# abs_area: 1803.7274537756666
# dist:3000000000562.91
# poly1 area 192851.5  poly2 area 199534.5 inter area 182063
# abs_area: 0.6055698577228437
# poly1 area 45436.5  poly2 area 199242.0 inter area 42702
# abs_area: 233.22485259736658
# poly1 area 7499.5  poly2 area 199534.5 inter area 7117
# abs_area: 2359.522296020159
# dist:9000000001295.844
# poly1 area 192851.5  poly2 area 108569.0 inter area 108099
# abs_area: 6.156243760347599
# poly1 area 192851.5  poly2 area 9658.0 inter area 9427
# abs_area: 261.4578044474137
# poly1 area 192851.5  poly2 area 1596.0 inter area 1401
# abs_area: 12373.471590289098
# poly1 area 45436.5  poly2 area 108569.0 inter area 40484
# abs_area: 71.78452609851746
# poly1 area 45436.5  poly2 area 9658.0 inter area 6752
# abs_area: 262.06263463606075
# poly1 area 45436.5  poly2 area 1596.0 inter area 1059
# abs_area: 980.1539268205402
# poly1 area 7499.5  poly2 area 108569.0 inter area 7117
# abs_area: 1076.1569270432328
# poly1 area 7499.5  poly2 area 9658.0 inter area 1978
# abs_area: 72.71656449659102
# poly1 area 7499.5  poly2 area 1596.0 inter area 223
# abs_area: 466.43264985365767
# dist:949.463590812256
# poly1 area 192851.5  poly2 area 107682.5 inter area 106760
# abs_area: 5.20911351687061
# poly1 area 192851.5  poly2 area 19401.5 inter area 19066
# abs_area: 87.00417679600706
# poly1 area 192851.5  poly2 area 622.0 inter area 538
# abs_area: 4597.095553013582
# poly1 area 45436.5  poly2 area 107682.5 inter area 38902
# abs_area: 76.7604919844339
# poly1 area 45436.5  poly2 area 19401.5 inter area 9612
# abs_area: 112.00035859540627
# poly1 area 45436.5  poly2 area 622.0 inter area 538
# abs_area: 10022.9087645529
# poly1 area 7499.5  poly2 area 107682.5 inter area 7117
# abs_area: 1071.9650508334619
# poly1 area 7499.5  poly2 area 19401.5 inter area 3167
# abs_area: 142.99054545109678
# poly1 area 7499.5  poly2 area 622.0 inter area 185
# abs_area: 1099.0209841717306
# dist:1412.0132157111052
# poly1 area 192851.5  poly2 area 1248.0 inter area 1122
# abs_area: 7607.5858749893305
# poly1 area 192851.5  poly2 area 3329.0 inter area 3141
# abs_area: 1290.801332498654
# poly1 area 192851.5  poly2 area 42214.5 inter area 41579
# abs_area: 45.7175760729602
# poly1 area 192851.5  poly2 area 21842.0 inter area 21836
# abs_area: 70.17515432506409
# poly1 area 45436.5  poly2 area 1248.0 inter area 1122
# abs_area: 2740.559414475756
# poly1 area 45436.5  poly2 area 3329.0 inter area 1799
# abs_area: 1256.5027782888533
# poly1 area 45436.5  poly2 area 42214.5 inter area 14977
# abs_area: 49.861062421536715
# poly1 area 45436.5  poly2 area 21842.0 inter area 13412
# abs_area: 92.69938963395721
# poly1 area 7499.5  poly2 area 1248.0 inter area 842
# abs_area: 352.0303838771214
# poly1 area 7499.5  poly2 area 3329.0 inter area 340
# abs_area: 181.09069727487184
# poly1 area 7499.5  poly2 area 42214.5 inter area 4887
# abs_area: 285.6399946794517
# poly1 area 7499.5  poly2 area 21842.0 inter area 5647
# abs_area: 110.77415849432228
# dist:2250000001264.7354
# poly1 area 192851.5  poly2 area 108913.5 inter area 107271
# abs_area: 5.049900303748961
# poly1 area 192851.5  poly2 area 17006.0 inter area 16680
# abs_area: 72.51343548031218
# poly1 area 192851.5  poly2 area 405.0 inter area 336
# abs_area: 14991.026215051763
# poly1 area 45436.5  poly2 area 108913.5 inter area 37132
# abs_area: 84.2894104528558
# poly1 area 45436.5  poly2 area 17006.0 inter area 8936
# abs_area: 123.23323160065895
# poly1 area 45436.5  poly2 area 405.0 inter area 336
# abs_area: 13168.245616725007
# poly1 area 7499.5  poly2 area 108913.5 inter area 7117
# abs_area: 1103.7414256473437
# poly1 area 7499.5  poly2 area 17006.0 inter area 2952
# abs_area: 122.05305727214119
# poly1 area 7499.5  poly2 area 405.0 inter area 135
# abs_area: 1840.0985500572258
# dist:2156.24836045702
# poly1 area 192851.5  poly2 area 61784.5 inter area 59784
# abs_area: 11.082662786665484
# poly1 area 192851.5  poly2 area 1145.0 inter area 990
# abs_area: 9858.407466011071
# poly1 area 192851.5  poly2 area 2050.0 inter area 1910
# abs_area: 2625.4317960730664
# poly1 area 192851.5  poly2 area 1898.0 inter area 1689
# abs_area: 8953.158985940447
# poly1 area 192851.5  poly2 area 1343.0 inter area 1170
# abs_area: 9665.065781921934
# poly1 area 192851.5  poly2 area 50002.5 inter area 48581
# abs_area: 10.193498903489113
# poly1 area 45436.5  poly2 area 61784.5 inter area 20644
# abs_area: 69.36503367695154
# poly1 area 45436.5  poly2 area 1145.0 inter area 990
# abs_area: 3334.6362154915996
# poly1 area 45436.5  poly2 area 2050.0 inter area 1910
# abs_area: 2319.649724005454
# poly1 area 45436.5  poly2 area 1898.0 inter area 1525
# abs_area: 1413.4520038010567
# poly1 area 45436.5  poly2 area 1343.0 inter area 1114
# abs_area: 2463.5611501241133
# poly1 area 45436.5  poly2 area 50002.5 inter area 15671
# abs_area: 62.01787214051615
# poly1 area 7499.5  poly2 area 61784.5 inter area 6997
# abs_area: 513.1756778107223
# poly1 area 7499.5  poly2 area 1145.0 inter area 806
# abs_area: 347.2429470537233
# poly1 area 7499.5  poly2 area 2050.0 inter area 1245
# abs_area: 183.69581812583684
# poly1 area 7499.5  poly2 area 1898.0 inter area 1301
# abs_area: 179.17408076649727
# poly1 area 7499.5  poly2 area 1343.0 inter area 908
# abs_area: 287.104262416847
# poly1 area 7499.5  poly2 area 50002.5 inter area 5826
# abs_area: 392.193453370745
# dist:6750000005959.7
# poly1 area 192851.5  poly2 area 3404.5 inter area 3150
# abs_area: 2444.6015530821714
# poly1 area 192851.5  poly2 area 12407.5 inter area 12162
# abs_area: 962.2612832000997
# poly1 area 192851.5  poly2 area 4920.0 inter area 4760
# abs_area: 836.2906682074606
# poly1 area 192851.5  poly2 area 2422.5 inter area 2311
# abs_area: 1235.9314070100327
# poly1 area 192851.5  poly2 area 163852.5 inter area 125147
# abs_area: 1.3804204966433073
# poly1 area 45436.5  poly2 area 3404.5 inter area 3091
# abs_area: 845.0488330893286
# poly1 area 45436.5  poly2 area 12407.5 inter area 8445
# abs_area: 191.45450052412374
# poly1 area 45436.5  poly2 area 4920.0 inter area 4615
# abs_area: 647.5438072553883
# poly1 area 45436.5  poly2 area 2422.5 inter area 2205
# abs_area: 1654.3586364527953
# poly1 area 45436.5  poly2 area 163800.5 inter area 29910
# abs_area: 164.09839477698236
# poly1 area 7499.5  poly2 area 3404.5 inter area 1077
# abs_area: 107.47303310365274
# poly1 area 7499.5  poly2 area 12407.5 inter area 2879
# abs_area: 24.467298035620708
# poly1 area 7499.5  poly2 area 4920.0 inter area 1489
# abs_area: 84.69701811179604
# poly1 area 7499.5  poly2 area 2422.5 inter area 675
# abs_area: 234.6281795591312
# poly1 area 7499.5  poly2 area 163918.0 inter area 7117
# abs_area: 1889.2218111879913
# dist:4500000001753.991
# poly1 area 192851.5  poly2 area 245484.0 inter area 175388
# abs_area: 8.022488606891656
# poly1 area 45436.5  poly2 area 245231.5 inter area 40975
# abs_area: 332.37456764205905
# poly1 area 7499.5  poly2 area 245484.0 inter area 7117
# abs_area: 2969.8750831687407
# dist:9000000001642.695
# poly1 area 192851.5  poly2 area 276550.0 inter area 190485
# abs_area: 4.622361808351326
# poly1 area 45436.5  poly2 area 276555.5 inter area 42083
# abs_area: 339.1579295796941
# poly1 area 7499.5  poly2 area 276847.0 inter area 7117
# abs_area: 3499.1410357585873
# dist:9000000001854.203
# poly1 area 192851.5  poly2 area 216859.5 inter area 169977
# abs_area: 3.8303032080404793
# poly1 area 45436.5  poly2 area 216572.0 inter area 40614
# abs_area: 238.7229687417724
# poly1 area 7499.5  poly2 area 216859.5 inter area 7117
# abs_area: 2582.5132049662398
# dist:9000000001438.625
# poly1 area 192851.5  poly2 area 38289.0 inter area 38280
# abs_area: 51.036782032836264
# poly1 area 192851.5  poly2 area 29817.0 inter area 29815
# abs_area: 39.39465804299693
# poly1 area 45436.5  poly2 area 38289.0 inter area 17597
# abs_area: 41.17071528247038
# poly1 area 45436.5  poly2 area 29817.0 inter area 16253
# abs_area: 52.66784262267609
# poly1 area 7499.5  poly2 area 38289.0 inter area 3648
# abs_area: 284.75577847472675
# poly1 area 7499.5  poly2 area 29817.0 inter area 6578
# abs_area: 154.79720406528634
# dist:3000000000269.5107
# poly1 area 192851.5  poly2 area 19734.0 inter area 19237
# abs_area: 467.29898863098344
# poly1 area 192851.5  poly2 area 18729.5 inter area 18719
# abs_area: 167.24204889616954
# poly1 area 192851.5  poly2 area 29068.0 inter area 28250
# abs_area: 164.20360454712005
# poly1 area 45436.5  poly2 area 19734.0 inter area 10424
# abs_area: 94.17294614276852
# poly1 area 45436.5  poly2 area 18729.5 inter area 14709
# abs_area: 118.12805891817452
# poly1 area 45436.5  poly2 area 29056.0 inter area 21130
# abs_area: 31.74628033652429
# poly1 area 7499.5  poly2 area 19734.0 inter area 2134
# abs_area: 31.89434459855929
# poly1 area 7499.5  poly2 area 18729.5 inter area 6419
# abs_area: 69.01407158385535
# poly1 area 7499.5  poly2 area 29068.0 inter area 6195
# abs_area: 144.59484792672814
# dist:292.36528203154916
# poly1 area 192851.5  poly2 area 2408.5 inter area 2252
# abs_area: 1935.3998740494146
# poly1 area 192851.5  poly2 area 2906.5 inter area 2708
# abs_area: 1733.8060895204176
# poly1 area 192851.5  poly2 area 8635.5 inter area 8268
# abs_area: 501.21018648311286
# poly1 area 192851.5  poly2 area 39060.0 inter area 37651
# abs_area: 42.97778457431513
# poly1 area 192851.5  poly2 area 103238.5 inter area 99459
# abs_area: 4.1448110983424895
# poly1 area 45436.5  poly2 area 2408.5 inter area 2252
# abs_area: 1669.3986707864397
# poly1 area 45436.5  poly2 area 2906.5 inter area 2560
# abs_area: 1348.6093265721925
# poly1 area 45436.5  poly2 area 8635.5 inter area 7033
# abs_area: 326.2503272286359
# poly1 area 45436.5  poly2 area 39060.0 inter area 12074
# abs_area: 60.43903112818466
# poly1 area 45436.5  poly2 area 103238.5 inter area 31142
# abs_area: 100.85516197341612
# poly1 area 7499.5  poly2 area 2408.5 inter area 1639
# abs_area: 125.17744874700708
# poly1 area 7499.5  poly2 area 2906.5 inter area 1927
# abs_area: 83.74612218861387
# poly1 area 7499.5  poly2 area 8635.5 inter area 4226
# abs_area: 30.127086195213185
# poly1 area 7499.5  poly2 area 39060.0 inter area 4241
# abs_area: 347.4529651833128
# poly1 area 7499.5  poly2 area 103238.5 inter area 7117
# abs_area: 1041.368160440791
# dist:4500000001252.377
# poly1 area 192851.5  poly2 area 2408.5 inter area 2252
# abs_area: 1935.3998740494146
# poly1 area 192851.5  poly2 area 2906.5 inter area 2708
# abs_area: 1733.8060895204176
# poly1 area 192851.5  poly2 area 8635.5 inter area 8268
# abs_area: 501.21018648311286
# poly1 area 192851.5  poly2 area 39060.0 inter area 37651
# abs_area: 42.97778457431513
# poly1 area 192851.5  poly2 area 103238.5 inter area 99459
# abs_area: 4.1448110983424895
# poly1 area 192851.5  poly2 area 66997.5 inter area 66003
# abs_area: 5.561529016846131
# poly1 area 192851.5  poly2 area 68430.0 inter area 67617
# abs_area: 4.303147346192362
# poly1 area 192851.5  poly2 area 47189.5 inter area 47178
# abs_area: 29.37152621980288
# poly1 area 45436.5  poly2 area 2408.5 inter area 2252
# abs_area: 1669.3986707864397
# poly1 area 45436.5  poly2 area 2906.5 inter area 2560
# abs_area: 1348.6093265721925
# poly1 area 45436.5  poly2 area 8635.5 inter area 7033
# abs_area: 326.2503272286359
# poly1 area 45436.5  poly2 area 39060.0 inter area 12074
# abs_area: 60.43903112818466
# poly1 area 45436.5  poly2 area 103238.5 inter area 31142
# abs_area: 100.85516197341612
# poly1 area 45436.5  poly2 area 66997.5 inter area 21065
# abs_area: 78.76203746398139
# poly1 area 45436.5  poly2 area 68394.0 inter area 21870
# abs_area: 74.0086418105452
# poly1 area 45436.5  poly2 area 47189.5 inter area 20584
# abs_area: 37.43551357721483
# poly1 area 7499.5  poly2 area 2408.5 inter area 1639
# abs_area: 125.17744874700708
# poly1 area 7499.5  poly2 area 2906.5 inter area 1927
# abs_area: 83.74612218861387
# poly1 area 7499.5  poly2 area 8635.5 inter area 4226
# abs_area: 30.127086195213185
# poly1 area 7499.5  poly2 area 39060.0 inter area 4241
# abs_area: 347.4529651833128
# poly1 area 7499.5  poly2 area 103238.5 inter area 7117
# abs_area: 1041.368160440791
# poly1 area 7499.5  poly2 area 66997.5 inter area 6712
# abs_area: 567.0714940460792
# poly1 area 7499.5  poly2 area 68430.0 inter area 7117
# abs_area: 545.2386584929288
# poly1 area 7499.5  poly2 area 47189.5 inter area 7117
# abs_area: 312.19500063214764
# dist:11250000001525.188
# poly1 area 192851.5  poly2 area 77388.0 inter area 75687
# abs_area: 7.499987574119829
# poly1 area 192851.5  poly2 area 9544.0 inter area 9274
# abs_area: 338.7523214425552
# poly1 area 192851.5  poly2 area 2881.0 inter area 2678
# abs_area: 879.1752248551524
# poly1 area 192851.5  poly2 area 36001.5 inter area 34773
# abs_area: 22.501285277659466
# poly1 area 192851.5  poly2 area 1339.0 inter area 1209
# abs_area: 5112.091215845278
# poly1 area 45436.5  poly2 area 77388.0 inter area 25893
# abs_area: 73.44440903876098
# poly1 area 45436.5  poly2 area 9544.0 inter area 8154
# abs_area: 241.53479640543148
# poly1 area 45436.5  poly2 area 2881.0 inter area 2634
# abs_area: 1579.519922664777
# poly1 area 45436.5  poly2 area 36001.5 inter area 10718
# abs_area: 87.36721545227813
# poly1 area 45436.5  poly2 area 1339.0 inter area 1209
# abs_area: 3169.2491455997347
# poly1 area 7499.5  poly2 area 77388.0 inter area 7117
# abs_area: 684.9267862192052
# poly1 area 7499.5  poly2 area 9544.0 inter area 4244
# abs_area: 33.195587923648596
# poly1 area 7499.5  poly2 area 2881.0 inter area 2065
# abs_area: 79.61841200712308
# poly1 area 7499.5  poly2 area 36001.5 inter area 3625
# abs_area: 325.75401449119664
# poly1 area 7499.5  poly2 area 1339.0 inter area 1063
# abs_area: 273.1623840496775
# dist:4500000001209.578
# poly1 area 192851.5  poly2 area 77388.0 inter area 75687
# abs_area: 7.499987574119829
# poly1 area 192851.5  poly2 area 9544.0 inter area 9274
# abs_area: 338.7523214425552
# poly1 area 192851.5  poly2 area 2881.0 inter area 2678
# abs_area: 879.1752248551524
# poly1 area 192851.5  poly2 area 36001.0 inter area 34773
# abs_area: 22.499646240540766
# poly1 area 192851.5  poly2 area 1339.0 inter area 1209
# abs_area: 5112.091215845278
# poly1 area 45436.5  poly2 area 77388.0 inter area 25893
# abs_area: 73.44440903876098
# poly1 area 45436.5  poly2 area 9544.0 inter area 8154
# abs_area: 241.53479640543148
# poly1 area 45436.5  poly2 area 2881.0 inter area 2634
# abs_area: 1579.519922664777
# poly1 area 45436.5  poly2 area 36001.0 inter area 10718
# abs_area: 87.37590578062623
# poly1 area 45436.5  poly2 area 1339.0 inter area 1209
# abs_area: 3169.2491455997347
# poly1 area 7499.5  poly2 area 77388.0 inter area 7117
# abs_area: 684.9267862192052
# poly1 area 7499.5  poly2 area 9544.0 inter area 4244
# abs_area: 33.195587923648596
# poly1 area 7499.5  poly2 area 2881.0 inter area 2065
# abs_area: 79.61841200712308
# poly1 area 7499.5  poly2 area 36001.0 inter area 3625
# abs_area: 325.7450366891593
# poly1 area 7499.5  poly2 area 1339.0 inter area 1063
# abs_area: 273.1623840496775
# dist:4500000001209.587
# poly1 area 192851.5  poly2 area 6512.0 inter area 6306
# abs_area: 275.59881154016756
# poly1 area 192851.5  poly2 area 188033.0 inter area 137672
# abs_area: 2.6460003317608827
# poly1 area 45436.5  poly2 area 6512.0 inter area 5575
# abs_area: 751.19865553038
# poly1 area 45436.5  poly2 area 188033.0 inter area 40766
# abs_area: 206.24156253280375
# poly1 area 7499.5  poly2 area 6512.0 inter area 2099
# abs_area: 54.27327347165795
# poly1 area 7499.5  poly2 area 188033.0 inter area 7117
# abs_area: 2337.49035527571
# dist:3000000000464.223