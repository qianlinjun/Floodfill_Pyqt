import json
import networkx as nx
import gmatch4py as gm
import matplotlib.pyplot as plt 

"""
Each algorithm is associated with an object, each object having its specific parameters. In this case, the parameters are the edit costs (delete a vertex, add a vertex, ...)
Each object is associated with a compare() function with two parameters. First parameter is a list of the graphs you want to compare, i.e. measure the distance/similarity (depends on the algorithm). Then, you can specify a sample of graphs to be compared to all the other graphs. To this end, the second parameter should be a list containing the indices of these graphs (based on the first parameter list). If you rather compute the distance/similarity between all graphs, just use the None value.
"""

# g1=nx.DiGraph()
# g1.add_edges_from([(0,3),(1,3),(2,3)])
# # print(g1.edges)
# g1.nodes[0]['type'] = 'Attack'
# g1.nodes[1]['type'] = 'Attack'
# g1.nodes[2]['type'] = 'Attack'
# g1.nodes[3]['type'] = 'Die'

# g2=nx.DiGraph()
# g2.add_edges_from([(0,3),(1,3),(2,3)])
# g2.nodes[0]['type'] = 'Die'
# g2.nodes[1]['type'] = 'Die'
# g2.nodes[2]['type'] = 'Die'
# g2.nodes[3]['type'] = 'Attack'


# g1 = nx.complete_bipartite_graph(5,4) 
# g2 = nx.complete_bipartite_graph(6,4)
# print(g1.nodes)
# print(g1.edges)

# nx.draw(g2, with_labels=True)                               #绘制网络G
# plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
# plt.show()   

# # node_del,node_ins,edge_del,edge_ins
# ged = gm.GraphEditDistance(1,1,1,1)
# # ged.set_attr_graph_used('type',None)

# # g1 -> g2 14.
# # g2 -> g1 10.
# # [[ 0. 14.]
# #  [10.  0.]]
# #n = len(listgs) return shape n * n 2 * 2
# result = ged.compare([g1,g2],None)
# print(result)



# max_=100
# size_g=10
# graphs_all = [nx.random_tree(size_g) for i in range(max_)]
# result_compiled=[]
# for size_ in tqdm(range(50,max_,50)):
#     graphs = graphs_all[:size_]
#     comparator=None
#     for class_ in [gm.BagOfNodes, gm.WeisfeleirLehmanKernel, gm.GraphEditDistance,  gm.GreedyEditDistance, gm.HED, gm.BP_2,  gm.Jaccard, gm.MCS, gm.VertexEdgeOverlap]:
#         deb=time.time()
#         if class_ in (gm.GraphEditDistance, gm.BP_2, gm.GreedyEditDistance, gm.HED):
#             comparator = class_(1, 1, 1, 1)
#         elif class_ == gm.WeisfeleirLehmanKernel:
#             comparator = class_(h=2)
#         else:
#             comparator=class_()
#         matrix = comparator.compare(graphs , None)
#         print([class_.__name__,size_,time.time()-deb])
#         result_compiled.append([class_.__name__,size_,time.time()-deb])

# df = pd.DataFrame(result_compiled,columns="algorithm size_data time_exec_s".split())
# df.to_csv("new_gmatch4py_res_{0}graphs_{1}size.csv".format(max_,size_g))



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




# G1 = nx.cycle_graph(6)
# nx.draw(G1, with_labels=True)                               #绘制网络G
# plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
# plt.show()   
# G2 = nx.wheel_graph(7)
# nx.draw(G2, with_labels=True)                               #绘制网络G
# plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
# plt.show()

dist = nx.graph_edit_distance(G, H, node_subst_cost=lambda x,y:1) #, node_subst_cost=lambda x,y:1
print("G H dist:", dist)
dist = nx.graph_edit_distance(G, K)
print("G K dist:", dist)



