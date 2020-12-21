import json
import networkx as nx
# import gmatch4py as gm
import matplotlib.pyplot as plt 


class SceneNode(object):
    def __init__(self):
        pass
    # build graph
    def compute_dist(G_node1, G_node2):
        pass

    def compute_angle(G_node1, G_node2):
        pass

    # calculate difference
    def compute_iou(G1_node, G2_node):
        pass

    def node_sub_cost(G1_node, G2_node):
        pass



class SceneGraph(object):
    def __init__(self):
        pass

    def buildGraphFromJson(self, json_file, visualise = False):
        # 两个地物是否是同一个类别
        # node edit cost
        # 山体 水 平原 森林 房屋
        # 同一个类别计算相似度
        scene_graph = nx.Graph()
        
        polygons = json.load(open(img1_res,'r'))
        
        

        # [1] add all nodes from image
        for idx, polygon in enumerate(polygons):
            id_     = polygon["id"]
            contour = polygon["contour"]
            area    = polygon["area"]
            momente = polygon["momente"]
            scene_graph.add_node(id_, contour=contour, area=area, momente=momente)
            for other_polygon in polygons[idx+1:]:
                other_polygon_id = other_polygon["id"]
                scene_graph.add_edge(id_, other_polygon_id)
        
        # [1] add global nodes
        # scene_graph.add_node("g", contour = contour, area=1024*1024, momente=momente)

        if visualise is True:
            nx.draw(scene_graph, with_labels=True)                               #绘制网络G
            plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
            plt.show()   

if __name__ == "__main__":    
    img1_res = "C:\qianlinjun\graduate\gen_dem\output\8.59262657_46.899601.json"
    buildGraphFromJson(img1_res, True)