import math
import numpy as np
# 定义一个节点类 Node，有横坐标和纵坐标两个属性，以及一个 label (0/1)
# index 为节点的编号，从 0 开始， 全局变量
class Node:
    def __init__(self, df, label, index):
        self.df = df
        self.label = label
        self.index = index
    
    def __str__(self):
        return str(self.index)

    def __repr__(self):
        return self.__str__()



# 给入的nodes是一个列表，每个元素是一个Node对象, distance是一个阈值
class Graph:
    def __init__(self, 
                 nodes: list,
                 distance=0.5):
        self.nodes = nodes
        self.distance = distance
        self.edges = []
        self.adjacency = {}
        self.degree = {}
        self.num_nodes = len(nodes)
        self.num_edges = 0
        self._init()
    
    # 初始化图的邻接表和度
    def _init(self):
        for i in range(self.num_nodes):
            self.adjacency[i] = []
            self.degree[i] = 0
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if distance(self.nodes[i], self.nodes[j]) < self.distance:
                    self.edges.append((i, j))
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)
                    self.degree[i] += 1
                    self.degree[j] += 1
                    self.num_edges += 1
    
    # 获取邻接矩阵
    def get_adjacency_matrix(self):
        matrix = [[0] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in self.adjacency[i]:
                matrix[i][j] = 1
        return matrix
    
    # 获取边
    def get_edges(self):
        return self.edges
    
    # 重写 __str__ 方法，用于打印图信息
    def __str__(self):
        return "Graph with {} nodes and {} edges".format(self.num_nodes, self.num_edges)
    
    #转为nx.Graph格式
    def to_nx(self):
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        return g
    

def distance(node1, node2):
    """
    Calculate the Euclidean distance between two nodes
    
    Parameters:
    - node1: The first node
    - node2: The second node
    
    Returns:
    - Euclidean distance
    """
    df1 = node1.df
    df2 = node2.df

    # Use numpy to calculate the Euclidean distance
    euclidean_distance = np.linalg.norm(df1 - df2)

    return euclidean_distance