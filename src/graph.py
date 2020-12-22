import math
import json
import networkx as nx
# import gmatch4py as gm
import matplotlib.pyplot as plt 


def compute_angle(G_node1, G_node2):
    '''
    return anti-clockwise raidus
    '''
    if G_node1.posXY[1] > G_node2.posXY[1]:
        return math.atan2(G_node1.posXY[1] - G_node2.posXY[1] , G_node2.posXY[0] - G_node1.posXY[0])
    else:
        return math.atan2(G_node2.posXY[1] - G_node1.posXY[1] , G_node1.posXY[0] - G_node2.posXY[0])

class SceneNode(object):
    def __init__(self, id, contour, pos, area):
        self.id      = id
        self.contour = contour
        self.posXY   = pos
        self.area    = area

    def compute_dist(self, node):
        return math.sqrt(self.posXY[1] - node.posXY[1] , node.posXY[0] - self.posXY[0])


    def compute_angle(self, node):
        '''
        return anti-clockwise raidus
        '''
        if self.posXY[1] > node.posXY[1]:
            return math.atan2(self.posXY[1] - node.posXY[1] , node.posXY[0] - self.posXY[0])
        else:
            return math.atan2(node.posXY[1] - self.posXY[1] , self.posXY[0] - node.posXY[0])

    # calculate difference
    def compute_iou(self, G1_node, G2_node):
        pass

    def node_sub_cost(self, G1_node, G2_node):
        pass



class SceneGraph(object):
    def __init__(self):
        pass

    def buildGraphFromJson(self, json_file, visualise = False):
        # 两个地物是否是同一个类别
        # node edit cost
        # 山体 水 平原 森林 房屋
        # 同一个类别计算相似度
        scene_graph = nx.Graph() # 建立一个空的无向图G
        
        polygons = json.load(open(img1_res,'r'))

        # [1] add all nodes from image
        for idx, polygon in enumerate(polygons):
            id_     = polygon["id"]
            contour = polygon["contour"]
            area    = polygon["area"]
            momente = polygon["momente"]
            scene_node = SceneNode(id_, contour, momente, area)
            scene_graph.add_node(id_, scene_node = scene_node )#contour=contour, area=area, momente=momente)
            for other_polygon in polygons[idx+1:]:
                id_1      = other_polygon["id"]
                contour_1 = other_polygon["contour"]
                area_1    = other_polygon["area"]
                momente_1 = other_polygon["momente"]
                scene_node_1 = SceneNode(id_1, contour_1, momente_1, area_1)
                angle = scene_node.compute_angle(scene_node_1)
                print("angle:{}".format(angle/3.1415926*180))
                scene_graph.add_edge(id_, id_1)
        
        # [1] add global nodes
        # scene_graph.add_node("g", contour = contour, area=1024*1024, momente=momente)

        if visualise is True:
            nx.draw(scene_graph, with_labels=True)                               #绘制网络G
            plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
            plt.show()   



if __name__ == "__main__":    
    scene_graph = SceneGraph()
    img1_res = "C:\qianlinjun\graduate\gen_dem\output\8.59262657_46.899601.json"
    scene_graph.buildGraphFromJson(img1_res, True)