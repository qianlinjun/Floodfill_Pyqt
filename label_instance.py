# encoding:utf8

import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np
from utils import fill_color_demo

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.widget = ImageWithMouseControl(self)
        self.widget.setGeometry(10, 10, 1200, 1200)
        self.setWindowTitle('Image with mouse control')



class PaintArea(QWidget):
    def __init__(self):
        super(PaintArea,self).__init__()
        self.left_top_point = None
        self.wait_paint_img     = None
        self.setMinimumSize(800,800)

    def set_pos_and_img(self, left_top_point, img):
        self.left_top_point = left_top_point
        self.wait_paint_img     = img

    def paintEvent(self, e):
        '''
        绘图
        :param e:
        :return:
        '''
        painter = QPainter(self)
        painter.begin(self)
        self.draw_img(painter)
        painter.end()
    
    def draw_img(self, painter):
        if self.left_top_point is None or self.wait_paint_img is None:
            print("draw img args: left_top_point or scale img is none")
            return
        print(type(self.wait_paint_img))
        if type(self.wait_paint_img) == QPixmap:
            painter.drawPixmap(self.left_top_point, self.wait_paint_img)
        else:
            print("QImage")
            painter.drawImage(self.left_top_point, self.wait_paint_img)
                    
        # self.left_top_point = None
        # self.scaled_img     = None


def cvimg_to_qtimg(cvimg):

    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

    return cvimg


class ImageWithMouseControl(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.img = None
        self.scaled_img = None
        self.cv_img = None
        self.left_top_point = QPoint(0, 0)
        self.initUI()

    def initUI(self):
        # left paint area
        spliter1 = QSplitter(Qt.Horizontal)
        spliter1.setOpaqueResize(True)
        
        stack1 = QStackedWidget()
        stack1.setFrameStyle(QFrame.Panel|QFrame.Raised)
        self.paintArea = PaintArea()
        stack1.addWidget(self.paintArea)    
        frame1 = QFrame(spliter1)
        layout_left = QVBoxLayout(frame1)
        #mainLayout1.setMargin(10)
        layout_left.setSpacing(6)
        layout_left.addWidget(stack1)

        # right button area
        spliter2 = QSplitter(Qt.Horizontal)
        # spliter2.setOpaqueResize(True)
        frame = QFrame(spliter2)
        gridLayout = QGridLayout(frame)
        load_btn = QPushButton()
        load_btn.clicked.connect(self.loadFile)
        load_btn.setText("open")
        gridLayout.addWidget(load_btn, 1, 0)
  
        self.setWindowTitle('Image with mouse control')

        full_layout = QGridLayout(self)
        full_layout.addWidget(spliter1,0,0)
        full_layout.addWidget(spliter2,0,1)
        self.setLayout(full_layout)

        # print("self.paintArea", self.paintArea.width(), self.paintArea.height())
    


    def loadFile(self):
        # load qpixmap
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'c:\\', 'Image files(*.jpg *.gif *.png)')
        self.img = QPixmap(fname)
        self.scaled_img = self.img#.scaled(self.img.size())
        # resize paint area
        self.paintArea.setMinimumSize(self.img.width(), self.img.height())
        
        # cv image to flood fill
        self.cv_img = cv2.imread(fname)
        gray2 = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        self.gray  = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)
        self.ori_img_wh = self.cv_img.shape

        self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
        self.repaint()

    def mouseMoveEvent(self, e):  # 重写移动事件
        if self.img is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in mouseMoveEvent ")
            return

        if self.left_click:
            self._endPos = e.pos() - self._startPos
            self.left_top_point = self.left_top_point + self._endPos
            self._startPos = e.pos()
            self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
            self.repaint()



    def mousePressEvent(self, e):
        if self.img is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in mousePressEvent ")
            return

        if e.button() == Qt.LeftButton:
            self.left_click = True
            self._startPos = e.pos()
            
            curPos = e.pos()
            # check valid window
            if curPos.x() <= 0 or curPos.x() >= self.paintArea.width() or curPos.y() <= 0 or curPos.y() >= self.paintArea.height():
                return  

            relat_pos = curPos - self.left_top_point
            if relat_pos.x() > 0 and relat_pos.x() < self.scaled_img.width() and relat_pos.y() > 0 and relat_pos.y() < self.scaled_img.height():
                # 得到鼠标点击位置在原始图片上的位置 作为漫水填充算法种子点
                point_map2_img_x = int(relat_pos.x()*1.0 / self.scaled_img.width() *  self.ori_img_wh[0])
                point_map2_img_y = int(relat_pos.y()*1.0 / self.scaled_img.height() *  self.ori_img_wh[1] )
                seedPt = (point_map2_img_x, point_map2_img_y)

                # fill_color_demo(self.gray, seedPt)

                copyImg = self.gray.copy()
                h, w = self.gray.shape[:2]
                mask = np.zeros([h+2, w+2], np.uint8)
                # mask 需要填充的位置设置为0
                cv2.floodFill(copyImg, mask, seedPt, (0, 0, 0), (5, 0, 0), (5, 0, 0))  #(836, 459) , cv2.FLOODFILL_FIXED_RANGE

                alpha = 0.5
                beta = 1-alpha
                gamma = 0

                # mask = np.vstack( (copyImg[:,:,np.newaxis], np.zeros([h, w, 2], np.uint8) ))
                mask = np.concatenate((copyImg[:,:,np.newaxis].astype(np.uint8), np.zeros([h, w, 2], dtype=np.uint8)), axis=-1)
                
                img_add = cv2.addWeighted(self.cv_img, alpha, mask, beta, gamma)# add mask
                qt_add_img = cvimg_to_qtimg(img_add)
                self.scaled_img = qt_add_img.scaled(self.scaled_img.size())
                
                self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
                self.repaint()



    def mouseReleaseEvent(self, e):
        if self.img is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in mouseReleaseEvent ")
            return

        if e.button() == Qt.LeftButton:
            self.left_click = False
        elif e.button() == Qt.RightButton:
            self.left_top_point = QPoint(0, 0)
            self.scaled_img = self.img.scaled(self.size())
            self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
            self.repaint()


    def wheelEvent(self, e):
        '''
        spin wheel to zoom in/out image
        '''
        if self.img is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in wheelEvent ")
            return

        if e.angleDelta().y() < 0:
            # 放大图片
            self.scaled_img = self.img.scaled(self.scaled_img.width()-15, self.scaled_img.height()-15)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.left_top_point.x())) / (self.scaled_img.width() + 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.left_top_point.y())) / (self.scaled_img.height() + 5)
            # print(new_w, new_h)
            self.left_top_point = QPoint(new_w, new_h)

        elif e.angleDelta().y() > 0:
            # 缩小图片
            self.scaled_img = self.img.scaled(self.scaled_img.width()+15, self.scaled_img.height()+15)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.left_top_point.x())) / (self.scaled_img.width() - 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.left_top_point.y())) / (self.scaled_img.height() - 5)
            self.left_top_point = QPoint(new_w, new_h)
        
        self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
        self.repaint()

    # def resizeEvent(self, e):
    #     if self.parent is not None and self.img is not None:
    #         self.scaled_img = self.img.scaled(self.size())
    #         self.left_top_point = QPoint(0, 0)
    #         self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    # ex = ImageWithMouseControl()
    ex.show()
    app.exec_()