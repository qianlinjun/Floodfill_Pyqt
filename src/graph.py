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



class SceneGraph(object):
    def __init__(self):
        pass

    def buildGraphFromJson(self, json_file, visualize = False):
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
                angle        = scene_node.compute_angle(scene_node_1)
                dist         = scene_node.compute_dist(scene_node_1)

                # print("{} -> {} angle:{}".format(id_, id_1, angle/3.1415926*180))
                scene_graph.add_edge(id_, id_1, name='{} -> {}'.format(id_, id_1), angle=angle, dist=dist)
        
        # [1] add global nodes
        # scene_graph.add_node("g", contour = contour, area=1024*1024, momente=momente)

        if visualize is True:
            pos = nx.spring_layout(scene_graph)
            # nx.draw(G, pos)
            nx.draw(scene_graph, pos, with_labels=True)#with_labels=True)                               #绘制网络G
            # node_labels = nx.get_node_attributes(G, 'desc')
            # nx.draw_networkx_labels(G, pos, labels=node_labels)
            edge_labels = nx.get_edge_attributes(scene_graph, 'name')
            nx.draw_networkx_edge_labels(scene_graph, pos, edge_labels=edge_labels)
            
            plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
            plt.show()   




def loadGraphsFromJson(save_path):
    scene_graphs = []
    for json_file in os.path.listdir(save_path):
        json_path = os.path.join(save_path, json_file)
        scene_graph = SceneGraph()
        # img1_res = "C:\qianlinjun\graduate\gen_dem\output\8.59262657_46.899601.json"
        scene_graph.buildGraphFromJson(json_path, visualize=True)
        scene_graphs.append(scene_graphs)
    
    return scene_graphs


def graph_index(query_G, scene_graphs):
    minimum_cost = 100000000000
    result_G = None
    for scene_G in scene_graphs:
        dist = nx.graph_edit_distance(query_G, scene_G)
        if minimum_cost > dist:
            minimum_cost = dist
            result_G     = scene_G

    print("search result:{}".format(minimum_cost))


if __name__ == "__main__":
    save_path = "C:\qianlinjun\graduate\gen_dem\output"
    scene_graphs = loadGraphsFromJson(save_path)
