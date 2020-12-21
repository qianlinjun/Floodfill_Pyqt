import json
import networkx as nx
# import gmatch4py as gm
import matplotlib.pyplot as plt 



# networks
G = nx.Graph()
H = nx.Graph()
K = nx.Graph()
G.add_node(0, a1=True)
G.add_node(1, a1=True)
G.add_edge(0,1)
# # nx.draw(G, with_labels=True)                               #绘制网络G
# # plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
# # plt.show()   


H.add_node("a", a2="Spam")
H.add_node("b", a2="Spam")
H.add_node("c", a2="Spam")
H.add_edges_from([("a","b"), ("a","c")])
# # nx.draw(H, with_labels=True)                               #绘制网络G
# # plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
# # plt.show()   

K.add_node("a", a2="Spam")
K.add_node("b", a2="Spam")
# K.add_node("c", a2="Spam")
K.add_edges_from([("a","b")])#, ("a","c")


# P = nx.tensor_product(G, H)#获得符合逻辑关系的交叉点
# print(nx.adjacency_matrix(P))
# # print(list(P))
# # [(0, 'a'), (1, 'a')]


# dist = nx.graph_edit_distance(G, H) #, node_subst_cost=lambda x,y:1
# print("G H dist:", dist)
# dist = nx.graph_edit_distance(G, K)
# print("G K dist:", dist)



# 两个地物是否是同一个类别
# node edit cost
# 山体 水 平原 森林 房屋
# 同一个类别计算相似度
def buildGraphFromJson(json_file, visualise = False):
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