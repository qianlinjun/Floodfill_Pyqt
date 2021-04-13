import os
import sys
import cv2
import math
import json
import heapq
import time
import numpy as np
np.set_printoptions(suppress=True)

from pykml import parser
import networkx as nx
# import gmatch4py as gm
import matplotlib.pyplot as plt 

from poly_util import get_iou, get_dist

img_w , img_h = 1024, 1024


__console__=sys.stdout


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


class  Edit():
    def __init__(self, ori_cost, update_cost, G, vertex_path, edge_path):
        self.ori_cost = ori_cost
        self.ori_rank = None
        self.update_cost = update_cost
        self.update_rank = None
        self.G    = G
        self.vertex_path = vertex_path
        self.edge_path   = edge_path
    # def __lt__(self, other):
    #     if self.cost < other.cost:
    #         return True
    #     else:
    #         return False



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
    if len(polygons) == 0:
        return None

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

    # global_id      = "global" 
    # global_contour = [[[0,0]],[[0,1024]],[[1024,1024]],[[1024,0]]]
    # global_moment  = [total_moment_x/len(polygons),total_moment_y/len(polygons)]
    # area = img_w * img_h
    # global_node = SceneNode(global_id, global_contour, global_moment, area, nodeType = "global")
    # scene_graph.add_node(global_id, scene_node = global_node)
    # for other_polygon in polygons:
    #     id_1      = other_polygon["id"]
    #     contour_1 = other_polygon["contour"]
    #     area_1    = other_polygon["area"]
    #     momente_1 = other_polygon["momente"]
    #     object_node_1 = SceneNode(id_1, contour_1, momente_1, area_1)
    #     angle        = global_node.compute_angle(object_node_1)
    #     dist         = global_node.compute_dist(object_node_1)

    #     # print("{} -> {} angle:{}".format(id_, id_1, angle/3.1415926*180))
    #     scene_graph.add_edge(global_id, id_1, name='{} - {}'.format(id_, id_1), angle=angle, dist=dist)
    
    return scene_graph




def loadGraphsFromJson(save_path, test_name=None, visualize=False):
    scene_graphs = []
    for json_file in os.listdir(save_path):
        if "json" not in json_file:
            continue
        if test_name is not None and test_name not in json_file:
            continue
        
        
        json_path = os.path.join(save_path, json_file)

        # print("{} \n".format(json_path))

        # scene_graph = SceneGraph()
        # img1_res = "C:\qianlinjun\graduate\gen_dem\output\8.59262657_46.899601.json"
        scene_graph = buildGraphFromJson(json_path)
        if scene_graph is not None:
            scene_graphs.append(scene_graph)

        if visualize is True:
            # color_map = [[255, 0, 51], [0, 255, 255],[0, 255, 0], [255, 0, 255], [0,0,255]]
            color_map = ["pink", "Magenta", "green", "cyan", "orange"]
            DrawGraph(scene_graph, node_color = color_map)
    
    return scene_graphs

def update_cost(g1, g2, veterx_edit_path, Cost_matrix):
    '''
    根据得到的编辑路径，重新更新结果
    '''
    cost_total = 0
    cost_detail = []

    g1_nodes = list(g1.nodes)
    g2_nodes = list(g2.nodes)
    for e in veterx_edit_path:
        n1, n2 = e
        cost = 0
        if n1 == None:
            #insert n2
            cost = math.sqrt( g2.nodes[n2]["scene_node"].area )    
        elif n2 == None:
            #del n1
            cost = math.sqrt( g1.nodes[n1]["scene_node"].area )
        else:
            # n1 substitute n2
            # Cost_matrix
            n1_row = g1_nodes.index(n1)
            n2_col = g2_nodes.index(n2)
            cost = Cost_matrix.C[n1_row, n2_col]
        
        cost_detail.append((e, cost))
        cost_total += cost
    # print("cost_detail", cost_detail)
    return cost_total, cost_detail




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
            # print("global node dist:{}".format(cost))
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
            # # print(poly1)
            poly2 = node2.contour
            poly1 = np.array(poly1).squeeze()
            poly2 = np.array(poly2).squeeze()
            # poly1_area, poly2_area, iou = get_iou(poly1, poly2)
            # iou_factor = 1. / (iou + 0.1)
            # humoments = cv2.matchShapes(poly1, poly2,  2, 0.0)
            # # print("insert hu moments:", cv2.matchShapes(poly1, np.array([[[0,0]]]),  2, 0.0))
            # area_factor = max(poly1_area, poly2_area) / min(poly1_area, poly2_area)
            # shape_cost =  humoments *  iou_factor * area_factor
            shape_cost = get_dist(poly1, node1.id, poly2, node2.id)
            

            #shaply

            #山脊线特征距离
            skyline_cost = 0

            cost = class_cost + shape_cost + skyline_cost
        
        return cost
    
    def node_del_cost(node1):
        # 当前节点和所有对方节点的替换损失最小值*0.9
        node1 = node1["scene_node"]
        return math.sqrt(node1.area)

    def node_ins_cost(node2):
        node2 = node2["scene_node"]
        return math.sqrt(node2.area)    

    def edge_subst_cost(edge1, edge2):
        abs_angle = abs(edge1["angle"] - edge2["angle"])/3.1415926 * 180 * 0.4 # 3.14
        abs_dist  = abs(edge1["dist"] - edge2["dist"]) * 0.2
        # print("edge cost:{}".format(abs_angle + abs_dist))
        return 0#abs_angle + abs_dist
    

    def edge_del_cost(edge1):
        
        return 0#edge1["dist"]*0.2

    def edge_ins_cost(edge2):
        return 0#edge2["dist"]*0.2


    minimum_cost = 100000000000
    result_G = None
    best_vertex_path = None 
    best_edge_path   = None
    
    cal_result = []
    for scene_G in scene_graphs:
        # 自定义替换损失
        # 自定义插入 删除损失
        
        ori_cost, vertex_path, edge_path, Cv, Ce = nx.graph_edit_distance(query_G, scene_G, 
                                                                        node_subst_cost = node_subst_cost, 
                                                                        node_del_cost = node_del_cost,
                                                                        node_ins_cost = node_ins_cost,
                                                                        edge_subst_cost = edge_subst_cost,
                                                                        edge_del_cost = edge_del_cost,
                                                                        edge_ins_cost = edge_ins_cost)

        cost_update, cost_detail = update_cost(query_G, scene_G, vertex_path, Cv)

        cal_result.append(Edit(ori_cost, cost_update, scene_G, vertex_path, edge_path))
        

        # heapq.nlargest()

        # print("dist:{}".format(ori_cost))
        if minimum_cost > cost_update:
            minimum_cost = cost_update
            result_G     = scene_G
            best_vertex_path = vertex_path
            best_edge_path   = edge_path
        
        print("G1: {} G2: {} dist:{}  edit_path{}".format(query_G.name.split("\\")[-1], scene_G.name.split("\\")[-1], ori_cost, vertex_path))
        print("update cost before:{} after:{} detail:{} \n\n".format(ori_cost, cost_update, cost_detail))
    

    # DrawGraph(result_G)
    
    print("search result:{} cost:{} vertex path:{} edge path:{} \n".format(result_G.name, minimum_cost, best_vertex_path, edge_path))

    # heapq.heapify(heap)
    # sort_result = heapq.nsmallest(50, heap)
    ori_sorted = sorted(cal_result, key = lambda edit : edit.ori_cost)
    for idx, edit in enumerate(ori_sorted):
        # print(item.G.name, item.cost)
        edit.ori_rank = idx
        ori_sorted[idx] = edit
        # rank_result.append(edit)
    
    update_sorted = sorted(ori_sorted, key = lambda edit : edit.update_cost)
    for idx, edit in enumerate(update_sorted):
        # print(item.G.name, item.cost)
        edit.update_rank = idx
        # update_sorted[idx] = edit
        print("{} {} ==> {}    {} ==> {}".format(edit.G.name, edit.ori_rank, edit.update_rank, edit.ori_cost, edit.update_cost))
    

# def load_meta(kmlPath):
#     meatDict = {}
#     with open(kmlPath, 'r') as f:
#         kml = parser.parse(f).getroot()
#         for pt in kml.Document.Placemark: # 遍历所有的Placemark Document.Placemark  
#             print(pt)
#             name = pt.name
#             coods = str(pt.Point.coordinates)

#             lon = float(coods.split(",")[0])
#             lat = float(coods.split(",")[1])

#             # heading = pt.LookAt.heading
#             meatDict


def test(result_save_dir):
    db_pic_w = 800
    db_pic_h = 800
    search_pic_w = 800
    search_pic_h = 800
    # data_path = "C:\qianlinjun\graduate\gen_dem\output\img"

    # data_path = "C:\qianlinjun\graduate\gen_dem\output\img_with_mask\switz-100-points"
    data_path = r"C:\qianlinjun\graduate\test-data\crop"
    scene_graphs = loadGraphsFromJson(data_path, visualize=False) #"8.59262657_46.899601"

    # query_path = r"C:\qianlinjun\graduate\test-data\query"
    # query_graphs = loadGraphsFromJson(query_path, visualize=False) #"8.59262657_46.899601"
    
    # R4
    query_path = r"C:\qianlinjun\graduate\test-data\query"
    query_graphs = loadGraphsFromJson(data_path, visualize=False) #"8.59262657_46.899601"

    if len(scene_graphs) <= 2:
        print("len(scene_graphs) <= 2")
        exit(0)
    

    for idx, G in enumerate(query_graphs):
        sys.stdout = __console__
        print("query_graph_id:", idx)

        queryGraph = query_graphs[idx]#scene_graphs[idx]
        queryName = queryGraph.name
        # lon lat
        queryLoc  = list(map(float, queryName.split("\\")[-1].replace(".json","").split("_")[1:]))
        queryResultName = queryName.split("\\")[-1].replace(".json","")
        queryResultFile = os.path.join(result_save_dir, "{}.txt".format(queryResultFile))

        fsock = open(queryResultFile, 'a+')
        sys.stdout = fsock
        print("\n\n-----------------------------------------\n")

        #*******# search
        GraphGallery =scene_graphs[0:idx] + scene_graphs[idx+1:]# scene_graphs #
        search_graph(queryGraph, GraphGallery)
        
        print("total: {} search: {} \n\n".format(len(scene_graphs), queryName))
        fsock.close()


        #*******# validation
        val_res = validate(queryLoc, queryResultFile)
        
        
        # if "11_8.53155708_46.60886" in G.name:
        
        # if "118_8.67111206_46.7680435" in G.name:
        # if "143_8.63542366_46.6530571" in G.name:
        # if "102_8.62759018_46.7402039" in G.name: no
        # if "183_8.74401855_46.6736794" in G.name: ok
        # if "15_8.56048775_46.6160736" in G.name: #ok
        # if "20_8.5722723_46.6212234" in G.name: no
        # if "30_8.59407043_46.639164" in G.name: ok
        # if "100_8.62820911_46.7310219" in G.name: 
        # if "110_8.64444923_46.7556992" in G.name: ok
        # if "120_8.66698265_46.7730026" in G.name: ok
        # if "130_8.72838211_46.6656761" in G.name: ok
        # if "150_8.67672729_46.6547318" in G.name: ok
        # if "2011-10-04_14.42.53_01024" in G.name:
        # if "dsc02737_01024" in G.name:ok
        # if "dsc02729_01024" in G.name: ok
        # if "2011-10-04_14.26.13_01024" in G.name: ok
        # if "183_8.74401855_46.6736794" in G.name:

        
        





if __name__ == "__main__":
    exper_name = "first"
    exper_folder = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) + "_" + exper_name
    exper_dir = os.path.join("C:\qianlinjun\graduate\src\src\output"  , exper_folder)
    if os.path.exists(exper_dir) is False:
        os.makedirs(exper_dir)
    
    test(exper_dir)
   

   