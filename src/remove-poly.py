# encoding:utf8
import os
import sys
import cv2
import json
from pathlib import Path

import numpy as np

from utils import Polygon, PaintArea, getOnePolygon, mergePoly, Operation, \
                  cvimg_to_qtimg, savePolygons2Json, loadPolygonFromJson


def loadJsonAndDraw(img_path, specify_poly_id=None):
    img_path = str(img_path)
    if "json" in img_path or "mask" in img_path:
        return

    
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print("cv_img is None")
        return
    
    json_path = img_path[:-4] + ".json"
    if os.path.exists(json_path) is False:
        return

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
        if specify_poly_id is not None and polygon.id != specify_poly_id:
            continue

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
    # cv2.imwrite(img_path.replace(".png","_mask.png"), img_add)
    # cv2.imshow("img_add", img_add)
    # # cv2.imshow("cv_img", cv_img)
    # cv2.waitKey()



def loadJsonAndFilter(json_path):
    print("input filter_polygon_ids:")
    filter_polygon_ids = list(map(int, sys.stdin.readline().strip().split(" ")))

    max_objectId, restore_polygons = loadPolygonFromJson(json_path) 

    filter_polygons = []
    for polygon in restore_polygons:
        if  polygon.id in filter_polygon_ids:
            continue
        filter_polygons.append(polygon)
    
    # savePolygons2Json(json_path.replace(".json", "_rep.json"), filter_polygons)
    # save_file_path = json_path.replace(".json", "_rep.json")
    print("save to {}".format(json_path))
    with open(json_path, "w") as save_f:
        list_2_save = []
        for poly in filter_polygons:
            if len(poly.contour) >= 3:
                cur_poly = {}
                # print("type id:{}".format(type(poly.id)))
                cur_poly["id"]        = poly.id
                # print("type contour:{}".format(type(poly.contour)))
                # print(poly.contour.tolist())
                cur_poly["contour"]   = poly.contour.tolist()
                # print("type fillColor:{}".format(type(poly.fillColor)))
                cur_poly["fillColor"] = poly.fillColor
                # print("type getArea:{}".format(type(poly.getArea())))
                cur_poly["area"]      = poly.getArea()
                # print("type getMoments:{}".format(type(poly.getMoments())))
                cur_poly["momente"]   = poly.getMoments()
                list_2_save.append(cur_poly)
        if len(list_2_save) > 0:
            json.dump(list_2_save, save_f, ensure_ascii=False)
            # save_f.write(data_2_save.encode('utf-8'))



def vis_all():
    img_dir = r"C:\qianlinjun\graduate\test-data\crop"
    for img_path in Path(img_dir).iterdir():
        print(img_path)
        loadJsonAndDraw(img_path)

def vis_poly():
    # img_path = r"C:\qianlinjun\graduate\data\switz-test-pts-3-17-11-image-fov-60\12_8.5369873_46.6108665.png"
    # specify_poly_id = 13
    print("input imgid:")
    img_id = sys.stdin.readline().strip()

    img_dir = r"C:\qianlinjun\graduate\test-data\crop"
    for json_path in Path(img_dir).iterdir():
        json_path = str(json_path)
        if "json" in json_path and img_id == json_path.split("\\")[-1].split("_")[0]:
            # print(img_path.split("\\")[-1].split("_")[0])
            loadJsonAndFilter(json_path)


if __name__ == '__main__':
    # vis_all()
    while True:
        vis_poly()