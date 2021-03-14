import os
import sys
import cv2
import math
import json
import numpy as np
import networkx as nx
# import gmatch4py as gm
import matplotlib.pyplot as plt 

from poly_util import get_iou

img_w , img_h = 1024, 1024


fsock = open('C:\qianlinjun\graduate\gen_dem\output\src\graph_match_out.txt', 'a+')
sys.stdout = fsock
print("\n\n-----------------------------------------\n")

def compute_angle(G_node1, G_node2):
    '''
    return anti-clockwise raidus
    '''
    if G_node1.posXY[1] > G_node2.posXY[1]:
        return math.atan2(G_node1.posXY[1] - G_node2.posXY[1] , G_node2.posXY[0] - G_node1.posXY[0])
    else:
        return math.atan2(G_node2.posXY[1] - G_node1.posXY[1] , G_node1.posXY[0] - G_node2.posXY[0])

class SceneNode(object):
    def __init__(self, id, contour, pos, area, nodeType=None, objectClass=None):
        self.id      = id 
        self.objectClass = objectClass
        self.nodeType    = nodeType
        self.contour = contour
        self.posXY   = pos
        self.area    = area

    def compute_dist(self, node):
        return math.sqrt(math.pow(self.posXY[1] - node.posXY[1],2) + math.pow(node.posXY[0] - self.posXY[0], 2) )


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



def DrawGraph(graph, node_color=None):
    pos = nx.spring_layout(graph)
    # nx.draw(G, pos)
    # nx.draw(graph, pos, with_labels=True)#with_labels=True)                               #绘制网络G
    # node_labels = nx.get_node_attributes(G, 'desc')
    # nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(graph, 'name')
    # if node_color is not None:
    #     nx.draw(graph, pos, with_labels=True, node_color=node_color,  node_size=1000, font_size=20) 
    #     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
    # else:
    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
    plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
    plt.show()   

# class SceneGraph(object):
#     def __init__(self):
#         pass

def buildGraphFromJson(json_file):
    # 两个地物是否是同一个类别
    # node edit cost
    # 山体 水 平原 森林 房屋
    # 同一个类别计算相似度
    scene_graph = nx.Graph(name=json_file) # 建立一个空的无向图G
    
    polygons = json.load(open(json_file,'r'))

    # [1] add object nodes from image
    total_moment_x = 0
    total_moment_y = 0
    for idx, polygon in enumerate(polygons):
        id_     = polygon["id"]
        contour = polygon["contour"]
        area    = polygon["area"]
        momente = polygon["momente"]
        object_node = SceneNode(id_, contour, momente, area, nodeType = "object", objectClass = "mountain")
        scene_graph.add_node(id_, scene_node = object_node )#contour=contour, area=area, momente=momente)
        total_moment_x += momente[0]
        total_moment_y += momente[1]
        
        for other_polygon in polygons[idx+1:]:
            id_1      = other_polygon["id"]
            contour_1 = other_polygon["contour"]
            area_1    = other_polygon["area"]
            momente_1 = other_polygon["momente"]
            object_node_1 = SceneNode(id_1, contour_1, momente_1, area_1)
            angle        = object_node.compute_angle(object_node_1)
            dist         = object_node.compute_dist(object_node_1)

            # print("{} -> {} angle:{}".format(id_, id_1, angle/3.1415926*180))
            scene_graph.add_edge(id_, id_1, name='{} - {}'.format(id_, id_1), angle=angle, dist=dist)
    
    # [2] add global nodes
    # print("len poly:{}".format(len(polygons)))
    global_id      = "global" 
    global_contour = [[[0,0]],[[0,1024]],[[1024,1024]],[[1024,0]]]
    global_moment  = [total_moment_x/len(polygons),total_moment_y/len(polygons)]
    area = img_w * img_h
    global_node = SceneNode(global_id, global_contour, global_moment, area, nodeType = "global")
    scene_graph.add_node(global_id, scene_node = global_node)
    for other_polygon in polygons:
        id_1      = other_polygon["id"]
        contour_1 = other_polygon["contour"]
        area_1    = other_polygon["area"]
        momente_1 = other_polygon["momente"]
        object_node_1 = SceneNode(id_1, contour_1, momente_1, area_1)
        angle        = global_node.compute_angle(object_node_1)
        dist         = global_node.compute_dist(object_node_1)

        # print("{} -> {} angle:{}".format(id_, id_1, angle/3.1415926*180))
        scene_graph.add_edge(global_id, id_1, name='{} - {}'.format(id_, id_1), angle=angle, dist=dist)
    
    return scene_graph




def loadGraphsFromJson(save_path, test_name=None, visualize=False):
    scene_graphs = []
    for json_file in os.listdir(save_path):
        if "json" not in json_file:
            continue
        if test_name is not None and test_name not in json_file:
            continue
        
        json_path = os.path.join(save_path, json_file)
        # scene_graph = SceneGraph()
        # img1_res = "C:\qianlinjun\graduate\gen_dem\output\8.59262657_46.899601.json"
        scene_graph = buildGraphFromJson(json_path)
        scene_graphs.append(scene_graph)

        if visualize is True:
            # color_map = [[255, 0, 51], [0, 255, 255],[0, 255, 0], [255, 0, 255], [0,0,255]]
            color_map = ["pink", "Magenta", "green", "cyan", "orange"]
            DrawGraph(scene_graph, node_color = color_map)
    
    return scene_graphs


# 两个地物是否是同一个类别
# node edit cost
# 山体 水 平原 森林 房屋
# 同一个类别计算相似度
# 归一化 到同一个尺度
def search_graph(query_G, scene_graphs):
    '''
    search graph accord minimum edit cost
    '''

    def node_subst_cost(node1, node2):
        node1 = node1["scene_node"]
        node2 = node2["scene_node"]
        # print(node1.id)

        #global 
        if node1.nodeType   == "global" and node2.nodeType == "global":
            # 位置距离
            # 整体的山脊线 差距
            cost = node1.compute_dist(node2)
            print("global global dist:{}".format(cost))
            # return dist
        elif node1.nodeType == "global" or  node2.nodeType == "global":
            # return 10000000000000
            cost = 10000000000000
            print("global node dist:{}".format(cost))
        else:
            # 普通节点  
            # 类别距离
            class_cost  = 0
            if node1.objectClass != node2.objectClass :
                class_cost = 10000000000000 # 10000000000000
            
            # dist
            # dist_cost = node1.compute_dist(node2)

            # area 像素误差
            # area_cost = abs(node1.area  - node2.area)
            
            # hu 矩具有
            poly1 = node1.contour
            poly2 = node2.contour
            poly1 = np.array(poly1).squeeze()
            poly2 = np.array(poly2).squeeze()
            poly1_area, poly2_area, iou = get_iou(poly1, poly2)
            shape_cost = cv2.matchShapes(poly1, poly2,  2, 0.0) *  1. / (iou + 0.1) * max(poly1_area, poly2_area) / min(poly1_area, poly2_area)
            print("node:{} node:{} shape_cost: {}".format(node1.id, node2.id, shape_cost))

            #shaply

            #山脊线特征距离
            skyline_cost = 0

            cost = class_cost + shape_cost + skyline_cost
        
        return cost
    
    def node_del_cost(node1):
        # 当前节点和所有对方节点的替换损失最小值*0.9
        
        return math.sqrt(node1.area)



    def node_ins_cost(node2):

        return math.sqrt(node2.area)
    

    def edge_subst_cost(edge1, edge2):
        # abs_angle = abs(edge1["angle"] - edge2["angle"])/3.1415926 * 180 # 3.14
        # abs_dist  = abs(edge1["dist"] - edge2["dist"])
        # print("edge cost:{}".format(abs_angle + abs_dist))
        return 0 #abs_angle + abs_dist
    

    def edge_del_cost(node1):
        return 0
    def edge_ins_cost(node1):
        return 0


    minimum_cost = 100000000000
    result_G = None
    best_vertex_path = None 
    best_edge_path   = None
    
    for scene_G in scene_graphs:
        # 自定义替换损失
        # 自定义插入 删除损失
        
        print("\n calculate dist between {} {}".format(query_G.name, scene_G.name))
        ge_dist, vertex_path, edge_path = nx.graph_edit_distance(query_G, scene_G, 
                                        node_subst_cost = node_subst_cost, 
                                        node_del_cost = node_del_cost,
                                        node_ins_cost = node_ins_cost,
                                        edge_subst_cost = edge_subst_cost,
                                        edge_del_cost = edge_del_cost,
                                        edge_ins_cost = edge_ins_cost)

        # print("dist:{}".format(ge_dist))
        if minimum_cost > ge_dist:
            minimum_cost = ge_dist
            result_G     = scene_G
            best_vertex_path = vertex_path
            best_edge_path   = edge_path
    
    DrawGraph(result_G)
    
    print("search result:{} cost:{} vertex path:{} edge path:{} \n".format(result_G.name, minimum_cost, best_vertex_path, edge_path))


if __name__ == "__main__":
    db_pic_w = 800
    db_pic_h = 800
    search_pic_w = 800
    search_pic_h = 800
    # save_path = "C:\qianlinjun\graduate\gen_dem\output\img"

    save_path = "C:\qianlinjun\graduate\gen_dem\output\img_with_mask\switz-100-points"
    scene_graphs = loadGraphsFromJson(save_path, visualize=False) #"8.59262657_46.899601"
    
    if len(scene_graphs) > 2:
        search_graph(scene_graphs[0], scene_graphs[1:])

    print("total: {} search: {} \n\n".format(len(scene_graphs), scene_graphs[0].name))
    fsock.close()
