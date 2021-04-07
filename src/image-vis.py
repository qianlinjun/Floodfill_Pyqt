# encoding:utf8

import sys
import cv2
from pathlib import Path

import numpy as np

from utils import Polygon, PaintArea, getOnePolygon, mergePoly, Operation, \
                  cvimg_to_qtimg, savePolygons2Json, loadPolygonFromJson


def loadJsonAndDraw(img_path):
    img_path = str(img_path)
    if "json" in img_path or "mask" in img_path:
        return

    
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print("cv_img is None")
        return
    
    json_path = img_path[:-4] + ".json"

    ori_img_hwc = cv_img.shape # h w c
    gray2 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)       
    gray  = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)
    copyImg = gray.copy()

    # ori_img_hwc[0]
    cur_mask = np.zeros([ori_img_hwc[0]+2, ori_img_hwc[1]+2], np.uint8)
    last_mask = cur_mask.copy()

    # if save file is exit then load it
    
    max_objectId, restore_polygons = loadPolygonFromJson(json_path) 

    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    # scaled_img_last_mask
    last_mask = np.zeros([ori_img_hwc[0]+2, ori_img_hwc[1]+2], np.uint8)

        
    dst = np.ones(cv_img.shape, dtype=np.uint8)*255

    for polygon in restore_polygons:
        #polygons_stack[::-1]:
        if len(polygon.contour) >= 3:
            # print("draw polygon area:{}".format(polygon.area))
            cv2.drawContours(dst, [polygon.contour], -1, polygon.fillColor, cv2.FILLED)# -1表示全画 (255, 0, 0)
            cv2.drawContours(last_mask, [polygon.contour], -1, 255, cv2.FILLED)# -1表示全画 (255, 0, 0)
            # font = cv2.FONT_HERSHEY_SITMPLEX
            cv2.putText(dst, str(polygon.id), polygon.getMoments(), 0, 1.2, (0, 0, 0), 2)
            # cv2.circle(dst, polygon.getMoments(), 5, polygon.fillColor, -1)   # 绘制圆点
        elif len(polygon.contour) == 1:
            cx = polygon.contour[0][0][0]
            cy = polygon.contour[0][0][1]
            cv2.circle(dst, (cx, cy), 5, polygon.fillColor, -1)   # 绘制圆点
        elif len(polygon.contour) == 2:
            x1 = polygon.contour[0][0][0]
            y1 = polygon.contour[0][0][1]
            x2 = polygon.contour[1][0][0]
            y2 = polygon.contour[1][0][1]
            cv2.line(dst, (x1, y1), (x2, y2), polygon.fillColor, 2)
            # cv2.circle(dst, (cx, cy), 2, (0, 0, 255), -1)   # 绘制圆点
        
    # if len(polygons) >= 2:
    #     mergePoly(ori_img_wh, polygons[0].contour, polygons[1].contour)

    img_add  = cv2.addWeighted(cv_img, alpha, dst, beta, gamma)# add mask
    cv2.imwrite(img_path.replace(".png","_mask.png"), img_add)
    # cv2.imshow("img_add", img_add)
    # cv2.waitKey()


if __name__ == '__main__':
    # img_path = r"C:\qianlinjun\graduate\data\switz-test-pts-3-17-11-image-fov-60\1_8.63674355_46.8405838.png"
    img_dir = r"C:\qianlinjun\graduate\data\switz-test-pts-3-17-11-image-fov-60"
    for img_path in Path(img_dir).iterdir():
        print(img_path)
        loadJsonAndDraw(img_path)
        