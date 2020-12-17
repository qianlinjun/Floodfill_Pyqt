# encoding:utf8

import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from utils import Polygon, getOnePolygon

# f_handler=open('C:\qianlinjun\graduate\gen_dem\output\out.log', 'w')
# sys.stdout=f_handler


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.widget = ImageWithMouseControl(self)
        self.widget.setGeometry(10, 10, 1200, 1200)
        self.setWindowTitle('Image with mouse control')



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




class ImageWithMouseControl(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        print("before init")
        self.parent = parent
        self.qImg = None
        self.scaled_img = None
        self.scaled_img_last_mask = None
        self.polygons = []# draw polygons in scaled mask
        self.polygons_undo_stack = []

        # value
        self.floodFillthread = 10
        # self.floodFillDowner = 5
        self.floodfillFlags = 4|(252<<8)|cv2.FLOODFILL_FIXED_RANGE
        self.drawMaskFlag = False
        
        self.cv_img = None
        self.left_top_point = QPoint(0,0) #只有在缩放和移动的时候才会改变
        self.initUI()

        self.right_click = False
        print("after init")

    def initUI(self):
        # left paint area
        spliter1 = QSplitter(Qt.Horizontal)
        spliter1.setOpaqueResize(True)
        frame1 = QFrame(spliter1)
        layout_left = QVBoxLayout(frame1)
        self.paintArea = PaintArea()
        self.paintArea.PosChangeSignal.connect(self.posChangeCallback)
        
        layout_left.setSpacing(6)
        layout_left.addWidget(self.paintArea)
        

        # right button area
        self.loadBtn = QPushButton()
        self.loadBtn.setMaximumSize(QSize(50, 50))
        self.loadBtn.clicked.connect(self.loadFile)
        self.loadBtn.setText("打开")

        self.spinbox = QSpinBox()
        self.spinbox.setMaximumSize(QSize(50, 20))
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(254)
        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setMinimum(1)
        self.sld.setMaximum(254)

        self.spinbox.valueChanged.connect(lambda: self.sld.setValue(self.spinbox.value()))
        self.sld.valueChanged.connect(self.changeFloodfillArgv)
        self.sld.setValue(self.floodFillthread)

        self.cbFlag = QCheckBox('固定差值')
        self.cbFlag.stateChanged.connect(self.changeFlag)
        self.cbFlag.setCheckState(Qt.Checked)

        self.cbShowMask = QCheckBox('show mask')
        self.cbShowMask.stateChanged.connect(self.changeDrawMaskFlag)
        self.cbShowMask.setCheckState(Qt.Checked)

        self.cbDrawPolygon = QCheckBox('draw by hand')
        # self.cbDrawPolygon.stateChanged.connect(self.ifDrawByHand)
        # self.cbFlag.setCheckState(Qt.Checked)

        self.posInfoLabel = QLabel()
        self.pixInfoLabel = QLabel()

        spliter2 = QSplitter(Qt.Horizontal)
        # spliter2.setOpaqueResize(True)
        frame = QFrame(spliter2)
        layoutRight = QGridLayout(frame)
        layoutRight.addWidget(self.loadBtn, 1, 0)
        layoutRight.addWidget(self.spinbox, 3, 0)
        layoutRight.addWidget(self.sld, 4, 0)
        layoutRight.addWidget(self.cbFlag, 5, 0)
        layoutRight.addWidget(self.cbShowMask, 6, 0)
        layoutRight.addWidget(self.cbDrawPolygon, 7, 0)
        layoutRight.addWidget(self.posInfoLabel,8, 0)
        layoutRight.addWidget(self.pixInfoLabel,9,0)
        

        fullLayout = QGridLayout(self)
        fullLayout.addWidget(spliter1,0,1)
        fullLayout.addWidget(spliter2,0,0)
        self.setLayout(fullLayout)

        # self.setWindowTitle('Image with mouse control')

    def loadFile(self):
        print("before load file")

        print("before QFileDialog")
        # load qpixmap
        fname, ret = QFileDialog.getOpenFileName(self, '选择图片', 'c:\\', 'Image files(*.jpg *.gif *.png)')
        if fname == "":
            print("no valid image")
            return
        print("after QFileDialog")

        self.qImg = QPixmap(fname)
        self.scaled_img = self.qImg#.scaled(self.qImg.size())
        # resize paint area
        self.paintArea.setMinimumSize(self.qImg.width(), self.qImg.height())
        
        # cv image to flood fill
        self.cv_img = cv2.imread(fname)
        gray2 = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        self.gray  = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)
        self.ori_img_wh = self.cv_img.shape
        
        self.polygons.clear()
        self.polygons_undo_stack.clear()

        self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
        self.repaint()

        print("after load file")
    
    def changeFloodfillArgv(self, v):
        self.spinbox.setValue(v)
        self.floodFillthread = v
    
    def changeFlag(self, e):
        if e > 0:
            self.floodfillFlags = 4 | (252<<8) | cv2.FLOODFILL_FIXED_RANGE
            self.spinbox.setValue(10)
        else:
            self.floodfillFlags = 4 | (252<<8)
            self.spinbox.setValue(5)
    
    def changeDrawMaskFlag(self, e):
        if e > 0:
            self.drawMaskFlag = True
        else:
            self.drawMaskFlag = False
        
        self.drawPolygonsOnCv()
        

    def mouseMoveEvent(self, e):  # 重写移动事件
        # print("before movement")
        if self.qImg is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in mouseMoveEvent ")
            return

        if self.right_click:
            print("mouseMoveEvent")
            self._endPos = e.pos() - self._startPos
            self.left_top_point = self.left_top_point + self._endPos
            self._startPos = e.pos()
            self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
            self.repaint()
        
        # print("after movement")



    def mousePressEvent(self, e):
        '''
        点击产生漫水填充算法
        '''

        # print("before mousePressEvent")
        if self.qImg is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in mousePressEvent ")
            return

        if e.button() == Qt.LeftButton:
            # self.left_click = True
            self._startPos = e.pos()
            getflag, polygon = self.floodFillOnce()#得到当前的polygon
            if getflag is True:
                self.polygons.append(polygon)#将当前polygon加入到list中
                self.polygons_undo_stack.clear()

                self.drawPolygonsOnCv()
        elif e.button() == Qt.RightButton:
            self.right_click = True
  
        # print("after mousePressEvent")
    

    def floodFillOnce(self):
        '''
        get segmentation by  mouse click
        '''
        flag, ptLocxy = self.getPtMapInImg(QPoint(self.mousePos.x(), self.mousePos.y()))
        if flag is True:
            # fill_color_demo(self.gray, seedPt)

            copyImg = self.gray.copy()
            h, w = self.gray.shape[:2]
            mask = np.zeros([h+2, w+2], np.uint8)

            
            # mask_fill = 175#252 #mask的填充值
            #floodFill充值标志
            # flags = 4|(252<<8)|cv2.FLOODFILL_FIXED_RANGE

            # mask 需要填充的位置设置为0
            cv2.floodFill(copyImg, mask, ptLocxy, 255, self.floodFillthread, self.floodFillthread, self.floodfillFlags)  #(836, 459) , cv2.FLOODFILL_FIXED_RANGE
            # cv2.imwrite("C:\qianlinjun\graduate\gen_dem\output\mask.png", mask)
            # cv2.waitKey(1000)

            # mask = np.vstack( (copyImg[:,:,np.newaxis], np.zeros([h, w, 2], np.uint8) ))
            mask = cv2.resize(mask, (h, w))
            # mask_rgb = np.concatenate((np.ones([h, w, 2], dtype=np.uint8),  mask[:,:,np.newaxis].astype(np.uint8) ), axis=-1)
            # mask_rgb[mask == 0] = [255,255,255]#[127,127,127]
            # mask_rgb[mask == 255] = [0,0,0]#[255,0,0]
            
            # cv2.imshow("mask_rgb", mask)
            # cv2.waitKey()

            self.scaled_img_mask = mask

            getflag, polygon = getOnePolygon(mask, ptLocxy)
            if getflag is True:
                return True, Polygon(polygon, len(self.polygons))
            else:
                return False, None
        else:
            return False, None

    def drawPolygonsOnCv(self):
        if self.cv_img is None:
            return

        alpha = 0.7
        beta = 1-alpha
        gamma = 0

        # scaled_img_last_mask


        if self.drawMaskFlag is True and len(self.polygons) > 0:
            dst = np.ones(self.cv_img.shape, dtype=np.uint8)
            for polygon in self.polygons[::-1]:
                # print("draw polygon area:{}".format(polygon.area))
                cv2.drawContours(dst, [polygon.contour], 0, polygon.fillColor, cv2.FILLED)#(255, 0, 0)

            img_add       = cv2.addWeighted(self.cv_img, alpha, dst, beta, gamma)# add mask
            qt_add_img    = cvimg_to_qtimg(img_add)
            scaled_qimg   = qt_add_img.scaled(self.scaled_img.size())
            scaled_Pixmap = QPixmap.fromImage(scaled_qimg)
        else:
            scaled_Pixmap   = self.scaled_img
            
        
        self.paintArea.set_pos_and_img(self.left_top_point, scaled_Pixmap)
        self.repaint()


    def mouseReleaseEvent(self, e):
        # print("before mouseReleaseEvent")
        if self.qImg is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in mouseReleaseEvent ")
            return

        if e.button() == Qt.LeftButton:
            self.left_click = False
        elif e.button() == Qt.RightButton:
            self.right_click = False
        # elif e.button() == Qt.RightButton:
        #     self.left_top_point = QPoint(0, 0)
        #     self.scaled_img = self.qImg.scaled(self.size())
        #     self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
        #     self.repaint()
        # print("after mouseReleaseEvent")


    def wheelEvent(self, e):
        '''
        spin wheel to zoom in/out image
        '''
        # print("before wheelEvent")

        if self.qImg is None or self.scaled_img is None:
            print("img is None or or scaled_img is None  in wheelEvent ")
            return

        if e.angleDelta().y() < 0:
            # 放大图片
            self.scaled_img = self.qImg.scaled(self.scaled_img.width()-15, self.scaled_img.height()-15)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.left_top_point.x())) / (self.scaled_img.width() + 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.left_top_point.y())) / (self.scaled_img.height() + 5)
            # print(new_w, new_h)
            self.left_top_point = QPoint(new_w, new_h)

            # self.drawPolygonsOnCv()

        elif e.angleDelta().y() > 0:
            # 缩小图片
            self.scaled_img = self.qImg.scaled(self.scaled_img.width()+15, self.scaled_img.height()+15)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.left_top_point.x())) / (self.scaled_img.width() - 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.left_top_point.y())) / (self.scaled_img.height() - 5)
            self.left_top_point = QPoint(new_w, new_h)

        
        self.drawPolygonsOnCv()


        # self.paintArea.set_pos_and_img(self.left_top_point, self.scaled_img)
        # self.repaint()
        # print("after wheelEvent")

    # def resizeEvent(self, e):
    #     if self.parent is not None and self.qImg is not None:
    #         self.scaled_img = self.qImg.scaled(self.size())
    #         self.left_top_point = QPoint(0, 0)
    #         self.update()




    def  posChangeCallback(self, x, y):
        '''
        receive mouse pos change signal 
        '''
        # print("before posChangeCallback")
        text = "x: {0},  y: {1}".format(x, y)
        self.mousePos = QPoint(x, y)
        self.posInfoLabel.setText(text)

        if self.cv_img is not None:
            flag, ptLocxy = self.getPtMapInImg(QPoint(x, y))
            if flag is True:
                pixel_value = self.cv_img[ptLocxy[1], ptLocxy[0]]
                text = "{0} {1} {2}".format(pixel_value[2], pixel_value[1], pixel_value[0])
                self.pixInfoLabel.setText(text)
        # print("after posChangeCallback")
    


    def getPtMapInImg(self, curPos):
        '''
        get mouse's point location mapped in image
        '''
        # print("before getPtMapInImg")
        # check valid window
        if curPos.x() <= 0 or curPos.x() >= self.paintArea.width() or curPos.y() <= 0 or curPos.y() >= self.paintArea.height():
            print("after getPtMapInImg False, (-1, -1)")
            return False, (-1, -1)

        if self.scaled_img is not None:
            relat_pos = curPos - self.left_top_point
            
            if relat_pos.x() > 0 and relat_pos.x() < self.scaled_img.width() and relat_pos.y() > 0 and relat_pos.y() < self.scaled_img.height():
                # 得到鼠标点击位置在原始图片上的位置 作为漫水填充算法种子点
                point_map2_img_x = int(relat_pos.x()*1.0 / self.scaled_img.width() *  self.ori_img_wh[0])
                point_map2_img_y = int(relat_pos.y()*1.0 / self.scaled_img.height() *  self.ori_img_wh[1] )
                # print(curPos, (self.left_top_point.x(), self.left_top_point.y()), (point_map2_img_x, point_map2_img_y))
                # print("after getPtMapInImg")
                return True, (point_map2_img_x, point_map2_img_y)
            else:
                print("after getPtMapInImg relat_pos.x() > 0 is false")
            
        return False, (-1, -1)

    
    def keyPressEvent(self,event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            # self.actionFile.save(self.action_text.toPlainText())
            # self.status.showMessage(self.action_text.toPlainText())#"保存成功 %s" % self.file)
            if event.key() == Qt.Key_Z:
                # ctrl z
                print("undo")
                if len(self.polygons) > 0:
                    polygon = self.polygons.pop()
                    self.polygons_undo_stack.append(polygon)

                
            elif event.key() == Qt.Key_Y:
                print("redo")
                if len(self.polygons_undo_stack) > 0:
                    polygon = self.polygons_undo_stack.pop()
                    self.polygons.append(polygon)

            
            self.drawPolygonsOnCv()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    # ex = ImageWithMouseControl()
    ex.show()
    app.exec_()