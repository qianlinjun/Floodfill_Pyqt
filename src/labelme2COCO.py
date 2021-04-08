import os
import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import numpy as np
import glob
import PIL.Image

REQUIRE_MASK = True

class labelme2coco(object):
    def __init__(self,labelme_json=[],save_json_path='./new.json'):
        '''
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        '''
        self.labelme_json=labelme_json
        self.save_json_path=save_json_path
        self.images=[]
        self.categories=[]
        self.annotations=[]
        # self.data_coco = {}
        self.label=[]
        self.annID=1
        self.height=0
        self.width=0
        self.require_mask = REQUIRE_MASK
        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            if not json_file == self.save_json_path:
                with open(json_file,'r') as fp:
                    data = json.load(fp)
                    image_path = json_file.replace(".json",".png")
                    if os.path.exists(image_path) is False:
                        continue

                    print(image_path)
                    self.images.append(self.image(image_path, num))

                    for shapes in data:
                        label= 1
    #                    if label[1] not in self.label:
                        if label not in self.label:
                            print("find new category: ")
                            self.categories.append(self.categorie(label))
                            print(self.categories)
                            # self.label.append(label[1])
                            self.label.append(label)
                        points = shapes['contour'] #list(np.array(shapes['contour']).reshape(-1)) #多个坐标转为
                        self.annotations.append(self.annotation(points,label,num))
                        self.annID+=1

    def image(self, img_path, img_id):
        image={}
        # img = utils.img_b64_to_arr(data['imageData'])

        # img=io.imread(data['imagePath'])
        img = cv2.imread(img_path, 0)
        height, width = img.shape[:2]
        img = None
        image['height']=height
        image['width'] = width
        image['id'] = img_id+1
        image['file_name'] = img_path.split('\\')[-1]

        self.height=height
        self.width=width

        return image

    def categorie(self,label):
        categorie={}
        categorie['supercategory'] = label
#        categorie['supercategory'] = label
        categorie['id']=len(self.label)+1
        categorie['name'] = label
#        categorie['name'] = label[1]
        return categorie

    def annotation(self,points,label,num):
        annotation={}
        # print(points)
        # x1 = points[0][0]
        # y1 = points[0][1]
        # x2 = points[1][0]
        # y2 = points[1][1]
        # contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) #points = [[x1, y1], [x2, y2]] for rectangle
        # contour = contour.astype(int)

        contour = np.array(points)
        area = cv2.contourArea(contour)
        print("contour is ", contour, " area = ", area)
        annotation['segmentation'] = [list(map(float, contour.flatten()))] #[list(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).flatten())]
            #
        annotation['iscrowd'] = 0
        annotation['area'] = area
        annotation['image_id'] = num+1

        if self.require_mask:
            # annotation['bbox'] = list(map(float,self.getbbox(points)))
            x, y, w, h = cv2.boundingRect(contour)
            annotation['bbox'] = [x, y, w, h]
        else:
            x1 = points[0][0]
            y1 = points[0][1]
            width = points[1][0] - x1
            height = points[1][1] - y1
            annotation['bbox']= list(np.asarray([x1, y1, width, height]).flatten())

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self,label):
        for categorie in self.categories:
#            if label[1]==categorie['name']:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getbbox(self,points):
        polygons = points
        mask = self.polygons_to_mask([self.height,self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x


        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r]

    def polygons_to_mask(self,img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco={}
        data_coco['images']=self.images
        data_coco['categories']=self.categories
        data_coco['annotations']=self.annotations
        return data_coco

    def save_json(self):
        print("in save_json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  



labelme_json=glob.glob('C:\qianlinjun\graduate\data\switz-test-pts-3-17-11-image-fov-60 - clean\*.json')
labelme2coco(labelme_json,r'C:\qianlinjun\graduate\src\src\train.json')