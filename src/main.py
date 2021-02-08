# encoding:utf8

import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from utils import Polygon, PaintArea, getOnePolygon, mergePoly, Operation, \
                  cvimg_to_qtimg, savePolygons2Json, loadPolygonFromJson

# f_handler=open('C:\qianlinjun\graduate\gen_dem\output\out.log', 'w')
# sys.stdout=f_handler

class InstanceLabelTool(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        print("before init")
        self.parent = parent
        self.qImg = None
        self.scaled_img = None
        self.scaled_img_last_mask = None
        self.operation_stack = []
        self.operation_undo_stack = []
        self.polygons_stack = []# draw polygons in scaled mask
        self.polygons_undo_stack = []
        self.operation_stack = []
        self.operation_undo_stack = []

        # value
        self.fname = None #图片路径
        # 4连通 填充颜色 FLOODFILL_FIXED_RANGE 为true 则填充和种子点像素相差floodFillthread 的点, 为false 则相邻点像素相差thread个值就填充
        self.floodfillFlags = 4|(252<<8)|cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        self.floodFillthread = 10 #
        self.showMaskFlag = False #是否显示mask
        self.showTmpMask = False  #展示中间结果
        self.drawByHand = False   #手绘标志
        self.handDrawing = False  #标记手绘是否结束

        self.objectId = -1
        
        self.cv_img = None
        self.left_top_point = QPoint(0,0) #只有在缩放和移动的时候才会改变
        self.initUI()

        self.right_click = False
        self.isDoubleClick = False
        print("after init")


    def initUI(self):
        # left paint area
        spliter1 = QSplitter(Qt.Horizontal)
        spliter1.setOpaqueResize(True)
        frame1 = QFrame(spliter1)
        layoutLeft = QVBoxLayout()#frame1
        self.paintArea = PaintArea()
        self.paintArea.PosChangeSignal.connect(self.posChangeCallback)
        
        layoutLeft.setSpacing(6)
        layoutLeft.addWidget(self.paintArea)
        

        # right button area
        self.loadBtn = QPushButton()
        self.loadBtn.setMaximumSize(QSize(50, 50))
        self.loadBtn.clicked.connect(self.loadFileCallBack)
        self.loadBtn.setText("open")
        self.loadBtn.setStyleSheet('''QPushButton{background:#A0D468;border-radius:5px;}''') #QPushButton:hover{background:yellow;}
        self.saveBtn = QPushButton()
        self.saveBtn.setMaximumSize(QSize(50, 50))
        self.saveBtn.clicked.connect(self.savePolygonsCallBack)
        self.saveBtn.setText("save")
        self.saveBtn.setStyleSheet('''QPushButton{background:#48CFAD;border-radius:5px;}''')

        self.spinbox = QSpinBox()
        self.spinbox.setMaximumSize(QSize(50, 20))
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(254)
        # self.sld = QSlider(Qt.Horizontal, self)
        # self.sld.setMinimum(1)
        # self.sld.setMaximum(254)

        self.spinbox.valueChanged.connect(self.switchFloodfillThread)#lambda: self.floodFillthread = self.spinbox.value()) #self.sld.setValue(self.spinbox.value()))
        # self.sld.valueChanged.connect(self.changeFloodfillArgv)
        # self.sld.setValue(self.floodFillthread)

        self.cbFlag = QCheckBox('固定差值')
        self.cbFlag.stateChanged.connect(self.switchFloodfillMode)
        self.cbFlag.setCheckState(Qt.Checked)

        self.cbShowMask = QCheckBox('show mask')
        self.cbShowMask.stateChanged.connect(self.switchShowMask)
        self.cbShowMask.setCheckState(Qt.Checked)

        self.cbShowTmpMask = QCheckBox('show_tmp_mask')
        self.cbShowTmpMask.stateChanged.connect(self.switchShowTmpMask)
        # self.cbFlag.setCheckState(Qt.Checked)

        self.cbDrawPolygon = QCheckBox('draw by hand')
        self.cbDrawPolygon.stateChanged.connect(self.switchDrawByHand)

        self.posInfoLabel = QLabel()
        # self.posInfoLabel.setGeometry(QRect(328, 240, 329, 27*4))
        self.posInfoLabel.setWordWrap(True)
        self.posInfoLabel.setAlignment(Qt.AlignTop)
        self.pixInfoLabel = QLabel("0 0 0")
        # self.pixInfoLabel.setGeometry(QRect(328, 240, 329, 27*4))
        self.pixInfoLabel.setWordWrap(True)
        self.pixInfoLabel.setAlignment(Qt.AlignTop)

        spliter2 = QSplitter(Qt.Horizontal)
        # spliter2.setOpaqueResize(True)
        frame = QFrame(spliter2)
        layoutRight = QVBoxLayout()#frame
        layoutRight.addWidget(self.loadBtn)#, 1, 0)
        layoutRight.addWidget(self.saveBtn)#, 9, 0)
        layoutRight.addWidget(self.spinbox)#, 2, 0)
        # layoutRight.addWidget(self.sld, 4, 0)
        layoutRight.addWidget(self.cbFlag)#, 3, 0)
        layoutRight.addWidget(self.cbShowMask)#, 4, 0)
        layoutRight.addWidget(self.cbDrawPolygon)#, 5, 0)
        layoutRight.addWidget(self.cbShowTmpMask)#, 6, 0)  
        layoutRight.addWidget(self.posInfoLabel)#,7, 0)
        layoutRight.addWidget(self.pixInfoLabel)#,8,0)
        
        # layoutRight("background-color:black;")
        

        # fullLayout = QGridLayout(self)
        # fullLayout.addWidget(spliter1,0,1)
        # fullLayout.addWidget(spliter2,0,0)
        fullLayout = QHBoxLayout()
        fullLayout.addLayout(layoutRight)
        fullLayout.addLayout(layoutLeft)
        self.setLayout(fullLayout)

        # self.setWindowTitle('Image with mouse control')

    def loadFileCallBack(self):

        # load qpixmap
        fname, ret = QFileDialog.getOpenFileName(self, '选择图片', 'c:\\', 'Image files(*.jpg *.gif *.png)')
        if fname == "":
            print("no valid image")
            return

        self.fname = fname
        print("fname:{}".format(fname))


        self.qImg = QPixmap(fname)
        self.scaled_img = self.qImg#.scaled(self.qImg.size())
        # resize paint area
        # self.paintArea.setMinimumSize(self.qImg.width(), self.qImg.height())
        
        # cv image to flood fill
        self.cv_img = cv2.imread(fname)
        self.ori_img_hwc = self.cv_img.shape # h w c
        gray2 = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)       
        self.gray  = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)
        self.copyImg = self.gray.copy()

        # self.ori_img_hwc[0]
        self.cur_mask = np.zeros([self.ori_img_hwc[0]+2, self.ori_img_hwc[1]+2], np.uint8)
        self.last_mask = self.cur_mask.copy()

        self.objectId = -1
        self.polygons_stack.clear()
        self.polygons_undo_stack.clear()

        # if save file is exit then load it
        json_file_path = fname[:-4] + ".json"
        max_objectId, restore_polygons = loadPolygonFromJson(json_file_path) 
        if max_objectId >= 0:
            self.polygons_stack = restore_polygons
            self.objectId = max_objectId
        
        # self.setWindowTitle(fname)
        self.cbDrawPolygon.setCheckState(Qt.Unchecked)#取消手动绘制
        self.drawPolygonsOnCv()



    def switchFloodfillThread(self, v):
        # self.spinbox.setValue(v)
        self.floodFillthread = v
    
    def switchFloodfillMode(self, e):
        if e > 0:
            self.floodfillFlags = 4 | (252<<8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
            self.spinbox.setValue(self.floodFillthread) #10
        else:
            self.floodfillFlags = 4 | (252<<8) | cv2.FLOODFILL_MASK_ONLY
            self.spinbox.setValue(self.floodFillthread) # 5
    
    def switchShowMask(self, e):
        if e > 0:
            self.drawMaskFlag = True
        else:
            self.drawMaskFlag = False
        
        self.drawPolygonsOnCv()
    
    def switchShowTmpMask(self, e):
        if e > 0:
            self.showTmpMask = True
        else:
            self.showTmpMask = False
    
    def switchDrawByHand(self, e):
        if e > 0:
            self.drawByHand = True
        else:
            self.drawByHand = False
    
    def savePolygonsCallBack(self, e):
        if self.fname is None:
            QMessageBox.information(self,"warning","no result to save",QMessageBox.Yes, QMessageBox.Yes) # |QMessageBox.No
            return

        savePolygons2Json(self.fname, self.polygons_stack)


        

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
            # self._startPos = e.pos()
            
            validRegion, ptLocxy = self.getPtMapInImg(QPoint(self.mousePos.x(), self.mousePos.y()))
            if validRegion is True:
                if self.drawByHand is True:
                    # draw poly by hand
                    curObject_Id = self.objectId + 1
                    curObject_Pos_In_stack = self.findPolygonByID(curObject_Id)
                    print("curObjectid:{}".format(curObject_Pos_In_stack))
                    
                    if curObject_Pos_In_stack >= 0:
                        # pass
                        self.polygons_stack[curObject_Pos_In_stack].addPoint(ptLocxy)
                        # add operation for undo
                        drawPolygonOpe = Operation(Operation.drawPolygonByHandType, curObject_Id, ptLocxy)
                        self.operation_stack.append(drawPolygonOpe)
                        self.operation_undo_stack.clear()
                    else:
                        # create a new poly
                        #双击的时候 object id 才增加
                        polygon = Polygon(np.array([[ptLocxy]]), curObject_Id)
                        self.polygons_stack.append(polygon)#将当前polygon加入到list中
                        self.polygons_undo_stack.clear()
                        # add operation for undo
                        insertPolygonOpe = Operation(Operation.insertPolygonType, curObject_Id, ptLocxy)
                        self.operation_stack.append(insertPolygonOpe)
                        self.operation_undo_stack.clear()
                        self.handDrawing = True

                    # self.pts_draw_byHand.append(ptLocxy)
                    self.drawPolygonsOnCv()
                else:
                    if self.handDrawing is True:
                        #没有绘制完上一个多边形
                        res = QMessageBox.question(self,"warning","last polygon is drawing, if finish it?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes) # 
                        if res == QMessageBox.Yes:
                            self.objectId += 1
                            self.handDrawing = False
                        else:
                            return

                    getflag, polygon = self.floodFillOnce(ptLocxy)#得到当前的polygon
                    if getflag is True:
                        
                        self.polygons_stack.append(polygon)#将当前polygon加入到list中
                        self.polygons_undo_stack.clear()
                        insertPolygonOpe = Operation(Operation.insertPolygonType, [polygon.id], ptLocxy)
                        self.operation_stack.append(insertPolygonOpe)
                        self.operation_undo_stack.clear()

                        self.drawPolygonsOnCv()
        elif e.button() == Qt.RightButton:
            self.right_click = True
  
        # print("after mousePressEvent")


    def mouseDoubleClickEvent(self,e):
        curObjectIdx = self.findPolygonByID(self.objectId + 1)
        if self.drawByHand is True and len(self.polygons_stack[curObjectIdx].contour) > 0:
            self.objectId += 1
            self.handDrawing = False


    def mouseDoubieCiickEvent(self, event):
#        if event.buttons () == QtCore.Qt.LeftButton:                           # 左键按下
#            self.setText ("双击鼠标左键的功能: 自己定义")
        # self.setText ("鼠标双击事件: 自己定义")
        print("double")

    def floodFillOnce(self, ptLocxy):
        '''
        get segmentation by  mouse click
        '''
        # flag, ptLocxy = self.getPtMapInImg(QPoint(self.mousePos.x(), self.mousePos.y()))
        # if flag is True:
        # fill_color_demo(self.gray, seedPt)

        # copyImg = self.gray.copy()
        # h, w = self.gray.shape[:2]
        mask = np.zeros([self.ori_img_hwc[0] + 2, self.ori_img_hwc[1] + 2], np.uint8)
        
        # mask_fill = 175#252 #mask的填充值
        #floodFill充值标志
        # flags = 4|(252<<8)|cv2.FLOODFILL_FIXED_RANGE

        # mask 需要填充的位置设置为0
        # self.last_mask = self.cur_mask
        self.cur_mask = self.last_mask.copy()
        # print("cur_mask {}".format(self.cur_mask.shape()))
        # self.cur_mask
        cv2.floodFill(self.copyImg, self.cur_mask, ptLocxy, 255, self.floodFillthread, self.floodFillthread, self.floodfillFlags)  #(836, 459) , cv2.FLOODFILL_FIXED_RANGE
        # cv2.imwrite("C:\qianlinjun\graduate\gen_dem\output\mask.png", mask)
        # cv2.waitKey(1000)
        # print(self.cur_mask.sum() , self.last_mask.sum())
        mask = self.cur_mask - self.last_mask
        # mask = np.vstack( (copyImg[:,:,np.newaxis], np.zeros([h, w, 2], np.uint8) ))
        mask = cv2.resize(mask, (self.ori_img_hwc[0], self.ori_img_hwc[1]))
        
        # mask_rgb = np.concatenate((np.ones([h, w, 2], dtype=np.uint8),  mask[:,:,np.newaxis].astype(np.uint8) ), axis=-1)
        # mask_rgb[mask == 0] = [255,255,255]#[127,127,127]
        # mask_rgb[mask == 255] = [0,0,0]#[255,0,0]
        
        if self.showTmpMask is True:            
            cv2.imshow("mask_rgb", mask)
            cv2.imshow("last_mask", self.last_mask)
            cv2.waitKey()

        self.scaled_img_mask = mask

        getflag, polygon = getOnePolygon(mask, ptLocxy)
        if getflag is True:
            self.objectId += 1
            return True, Polygon(polygon, self.objectId)#, len(self.polygons_stack)
        else:
            return False, None
        # else:
        #     return False, None
        

    def drawPolygonsOnCv(self):
        if self.cv_img is None:
            return

        alpha = 0.7
        beta = 1-alpha
        gamma = 0
        # scaled_img_last_mask
        self.last_mask = np.zeros([self.ori_img_hwc[0]+2, self.ori_img_hwc[1]+2], np.uint8)
        if self.drawMaskFlag is True and len(self.polygons_stack) > 0:
            
            dst = np.ones(self.cv_img.shape, dtype=np.uint8)*255

            for polygon in self.polygons_stack[::-1]:
                if len(polygon.contour) >= 3:
                    # print("draw polygon area:{}".format(polygon.area))
                    cv2.drawContours(dst, [polygon.contour], -1, polygon.fillColor, cv2.FILLED)# -1表示全画 (255, 0, 0)
                    cv2.drawContours(self.last_mask, [polygon.contour], -1, 255, cv2.FILLED)# -1表示全画 (255, 0, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(dst, str(polygon.id), polygon.getMoments(), font, 1.2, (0, 0, 0), 2)
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
            
            # if len(self.polygons) >= 2:
            #     mergePoly(self.ori_img_wh, self.polygons[0].contour, self.polygons[1].contour)

            img_add       = cv2.addWeighted(self.cv_img, alpha, dst, beta, gamma)# add mask
            qt_add_img    = cvimg_to_qtimg(img_add)
            scaled_qimg   = qt_add_img.scaled(self.scaled_img.size())
            scaled_Pixmap = QPixmap.fromImage(scaled_qimg)
        else:
            scaled_Pixmap   = self.scaled_img
        
        self.paintArea.set_pos_and_img(self.left_top_point, scaled_Pixmap)
        self.repaint()
    

    def findPolygonByID(self, id):
        for idx, polygon in enumerate(self.polygons_stack):
            if id == polygon.id:
                return idx
        return -1

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
            new_left = e.x() - (self.scaled_img.width() * (e.x() - self.left_top_point.x())) / (self.scaled_img.width() + 5)
            new_top = e.y() - (self.scaled_img.height() * (e.y() - self.left_top_point.y())) / (self.scaled_img.height() + 5)
            # print(new_w, new_h)
            self.left_top_point = QPoint(new_left, new_top)

            # self.drawPolygonsOnCv()

        elif e.angleDelta().y() > 0:
            # 缩小图片
            self.scaled_img = self.qImg.scaled(self.scaled_img.width()+15, self.scaled_img.height()+15)
            new_left = e.x() - (self.scaled_img.width() * (e.x() - self.left_top_point.x())) / (self.scaled_img.width() - 5)
            new_top = e.y() - (self.scaled_img.height() * (e.y() - self.left_top_point.y())) / (self.scaled_img.height() - 5)
            self.left_top_point = QPoint(new_left, new_top)

        
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
        if self.fname is not None:
            text = "{0} \n x: {1},  y: {2}".format(str(self.fname), x, y)
        else:
            text = "x: {0},  y: {1}".format(x, y)
        self.mousePos = QPoint(x, y)
        self.posInfoLabel.setText(text)
        self.posInfoLabel.adjustSize()

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
                point_map2_img_x = int(relat_pos.x()*1.0 / self.scaled_img.width() *  self.ori_img_hwc[1])
                point_map2_img_y = int(relat_pos.y()*1.0 / self.scaled_img.height() *  self.ori_img_hwc[0] )
                # print("self.scaled_img.width() {}   self.ori_img_hwc[1] {}".format(self.scaled_img.width(), self.ori_img_hwc[1]))
                # print("points", curPos, (self.left_top_point.x(), self.left_top_point.y()), (point_map2_img_x, point_map2_img_y))
                # print("after getPtMapInImg")
                return True, (point_map2_img_x, point_map2_img_y)
            else:
                print("after getPtMapInImg relat_pos.x() > 0 is false")
            
        return False, (-1, -1)

    
    def keyPressEvent(self,event):

        # print("hello",QApplication.keyboardModifiers() == Qt.ControlModifier, QApplication.keyboardModifiers())
        for modifiers in dir(Qt):
            if QApplication.keyboardModifiers() == getattr(Qt, modifiers):
                print("modifiers:", modifiers)

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            # self.actionFile.save(self.action_text.toPlainText())
            # self.status.showMessage(self.action_text.toPlainText())#"保存成功 %s" % self.file)
            print("hi")
            if event.key() == Qt.Key_Z:
                # undo
                if len(self.polygons_stack) > 0:
                    
                    last_operation = self.operation_stack.pop()
                    if last_operation.operateType == Operation.insertPolygonType:
                        polygon = self.polygons_stack.pop()
                        self.polygons_undo_stack.append(polygon)
                        self.handDrawing = False
                    elif last_operation.operateType == Operation.drawPolygonByHandType:
                        self.polygons_stack[-1].removeLastPoint()
                    elif last_operation.operateType == Operation.mergePolygonsType:
                        pass
                    self.operation_undo_stack.append(last_operation)
                
            elif event.key() == Qt.Key_Y:
                
                if len(self.operation_undo_stack) > 0:
                    last_operation = self.operation_undo_stack.pop()
                    if last_operation.operateType == Operation.insertPolygonType:
                        polygon = self.polygons_undo_stack.pop()
                        self.polygons_stack.append(polygon)
                    elif last_operation.operateType == Operation.drawPolygonByHandType:
                        self.polygons_stack[-1].addPoint([[last_operation.relatePos]])
                    elif last_operation.operateType == Operation.mergePolygonsType:
                        pass
                    self.operation_stack.append(last_operation)

            print("self.polygons_stack", len(self.polygons_stack))
            self.drawPolygonsOnCv()



class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.widget = InstanceLabelTool(self)
        self.widget.setGeometry(10, 10, 1200, 1200)
        self.setWindowTitle('image label tool')




if __name__ == '__main__':
    app = QApplication(sys.argv)
    print("click left button to segmention")
    ex = Main()
    # ex = ImageWithMouseControl()
    ex.show()
    app.exec_()