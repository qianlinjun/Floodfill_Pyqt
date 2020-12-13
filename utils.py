import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

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



def fill_color_demo(image, pos=(1,1)):
    copyIma = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    # mask 需要填充的位置设置为0
    cv2.floodFill(copyIma, mask, pos, (0, 0, 0), (5, 0, 0), (5, 0, 0))  #(836, 459) , cv2.FLOODFILL_FIXED_RANGE
    # cv.imshow("fill_color", copyIma)
    # plt.subplot(1,2,1)
    # plt.imshow(image),
    # plt.title('image')
    # plt.subplot(1,2,2)
    plt.imshow(copyIma),
    plt.title('copyIma')

    plt.show()
    # return mask

# fill_color_demo(gray)


