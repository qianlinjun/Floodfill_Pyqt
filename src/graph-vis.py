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
    if node_color is not None:
        nx.draw(graph, pos, with_labels=True, node_color=node_color,  node_size=1000, font_size=20) 
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
    else:
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

        m1 = cv2.moments(np.array(contour)) # m原始 mu 中心化 nu 归一化
        # print("m1", m1)
        hu = cv2.HuMoments( m1 )
        np.set_printoptions(precision = 4)
        print(id_, area, momente, hu.squeeze())
        
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
            print('{} - {}'.format(id_, id_1), round(angle/3.14*180, 1), round(dist, 1))

            # print("{} -> {} angle:{}".format(id_, id_1, angle/3.1415926*180))
            scene_graph.add_edge(id_, id_1, name='{} - {}'.format(id_, id_1), angle=angle, dist=dist)
    
    
    A=np.array(nx.adjacency_matrix(scene_graph).todense())
    print("adjacency_matrix", A)
    
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
            shape_cost = get_dist(poly1, node1.id, poly2, node2.id, verbose)
            # print(shape_cost)

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
        abs_angle = abs(edge1["angle"] - edge2["angle"])/3.1415926 * 180  #* 0.4 # 3.14
        print("absangle", edge1["angle"]/3.1415926 * 180, edge2["angle"]/3.1415926 * 180)
        abs_dist  = abs(edge1["dist"] - edge2["dist"]) * 0.3
        # print("edge cost:{} {}".format(abs_angle , abs_dist))
        return  abs_angle + abs_dist  
    
    def edge_del_cost(edge1):
        return edge1["dist"]* 0.3   #+ edge1["angle"]/3.1415926 * 180 * 3#*0.2

    def edge_ins_cost(edge2):
        return edge2["dist"]* 0.3 #+ edge2["angle"]/3.1415926 * 180 * 3#*0.2


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

        # cost_update, cost_detail = update_cost(query_G, scene_G, vertex_path, Cv)
        cost_update = ori_cost
        cal_result.append(Edit(ori_cost, cost_update, scene_G, vertex_path, edge_path))
        

        # heapq.nlargest()

        # print("dist:{}".format(ori_cost))
        if minimum_cost > ori_cost:
            minimum_cost = ori_cost
            result_G     = scene_G
            best_vertex_path = vertex_path
            best_edge_path   = edge_path
        if verbose is True:
            print("G1: {} G2: {} dist:{}  edit_path{}".format(query_G.name.split("\\")[-1], scene_G.name.split("\\")[-1], ori_cost, vertex_path))
            # print("update cost before:{} after:{} detail:{} \n\n".format(ori_cost, cost_update, cost_detail))
            print("update cost before:{} after:{} \n\n".format(ori_cost, cost_update))
    

    # DrawGraph(result_G)
    if verbose is True:
        print("search result:{} cost:{} vertex path:{} edge path:{} \n".format(result_G.name, minimum_cost, best_vertex_path, edge_path))

    # heapq.heapify(heap)
    # sort_result = heapq.nsmallest(50, heap)
    ori_sorted = sorted(cal_result, key = lambda edit : edit.ori_cost)
    for idx, edit in enumerate(ori_sorted):
        # print(item.G.name, item.cost)
        edit.ori_rank = idx
        ori_sorted[idx] = edit
        # rank_result.append(edit)
        # print("{} {}    {} ==> {}".format(edit.G.name, edit.ori_rank, edit.ori_cost))


    update_sorted = sorted(ori_sorted, key = lambda edit : edit.update_cost)
    for idx, edit in enumerate(update_sorted):
        # print(item.G.name, item.cost)
        edit.update_rank = idx
        # update_sorted[idx] = edit
        if verbose is True:
            print("{} {} ==> {}    {} ==> {}".format(edit.G.name, edit.ori_rank, edit.update_rank, edit.ori_cost, edit.update_cost))
    
    return ori_sorted
    



from math import radians, cos, sin, asin, sqrt
def meters_from_latlon(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

val_count = 0



def cal_dist(queryLoc, searchResut):
    # global val_count
    val_count = 0
    resDistList = []
    for idx, res in enumerate(searchResut):
        res_name = res.G.name.split("\\")[-1].replace(".json","")
        resLoc = res_name.split("_")[1:]
        # lon lat
        resLoc = list(map(float, resLoc))

        dist = meters_from_latlon(queryLoc[0], queryLoc[1], resLoc[0], resLoc[1])
        # print("dist", , queryLoc[0], queryLoc[1], resLoc[0], resLoc[1], dist)
        # print()
        resDistList.append(res_name+"|"+str(dist))
       

    return val_count, resDistList





def vis_graph():
    query_path = r"C:\qianlinjun\graduate\test-data\query"
    test_name = r"6_8.721228366639377_46.67035726569055"
    scene_graphs = loadGraphsFromJson(query_path, test_name, visualize=False) #"8.59262657_46.899601"
    


N = 25
def analysis_result(res_path):
    # with open(path, "r") as f:
    #     resList = f.readlines()
    #     for res in resList:
    #         res = res.strip()
    query_path = r"C:\qianlinjun\graduate\test-data\query"
    query_graphs = loadGraphsFromJson(query_path, visualize=False) #"8.59262657_46.899601"

    if len(query_graphs) <= 2:
        print("len(query_graphs) <= 2")
        exit(0)
    
    with open(res_path, "r") as f:
        allResList = f.readlines()
        # for res in allResList:
        #     res = res.strip().spilt(" ")

    
    # count = 0
    query_topN = []
    for g_idx, G in enumerate(query_graphs):
        if g_idx >= len(allResList):
            continue

        ifvalid = False
        print("analysis {}".format(G.name))
        resList = allResList[g_idx]
        resList = resList.strip().split(" ")
        
        topN_findlist = [0 for i in range(N)] # 0 5 10 15 20 25

        for res_idx, res in enumerate(resList):
            filename, ged_dist = res.split("|")
            ged_dist = float(ged_dist)
            for topN in range(N):
                # topN = n_idx * 5
                # if topN == 0:
                #     topN = 1
            
                if ged_dist < 1000 and res_idx <= topN :
                    topN_findlist[topN] += 1
                    # if ifvalid is False:
                    #     ifvalid = True
                    #     print("query{}".format(G.name))
                    query_loc = G.name.split("\\")[-1].replace(".json","").split("_")[1:3]
                    query_lon = round(float(query_loc[0]), 6)
                    query_lat = round(float(query_loc[1]), 6)
                    res_loc = filename.split("_")[1:]
                    res_lon = round(float(res_loc[0]), 6)
                    res_lat = round(float(res_loc[1]), 6)

                    # print( "{},{} {},{} {}".format(query_lon, query_lat, res_lon, res_lat, res_idx+1))
                    # break

        
        # if ifvalid is True:
        #     count += 1
        query_topN.append(topN_findlist)
        print(topN_findlist) 
        # print("\n")
    # print("valid query", count)

    query_topN = np.array(query_topN)

    
    print("recall_topN", [np.sum(query_topN[:, i]>0) for i in range(N)])

    return query_topN




if __name__ == "__main__":
    vis_graph()







   

   