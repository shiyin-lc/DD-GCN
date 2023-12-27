from collections import defaultdict

import numpy as np
from typing import Tuple, List

# num_nodes = 25
epsilon = 1e-6

# todo: max_hop要不要单独作为一个参数？
class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:

        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partitistrategy (string): must be one of the follow candidateson Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    # 把list变成可变长度的列表，生成不同的A，后面可以为STGCN构造不同尺度的Graph todo:改这里的graph还是改model里graph的参数
    def __init__(self,
                 layout='ntu-rgb+d',  # openpose or ntu-rgb+d  # 改
                 strategy='activity',  # uniform or distance or spatial  # 改
                 max_hop=1,  # the maximal distance between two connected nodes  # 这里要做成列表  todo:相当于分几类?到中心节点的最大距离
                 dilation=1, # controls the spacing between the kernel points  # todo:相当于label的间隔?
                 ):  # 图里面把max_hop作为尺度因子，改成 max_scale = 3
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)  # 获取graph边信息
        # 新增函数求原点和终点矩阵
        self.source_M, self.target_M, self.nor_source_graph, self.nor_target_graph = build_digraph_incidence_matrix(self.num_node,
                                                                                                                    self.edge)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)  # 返回的是距离矩阵


        self.get_adjacency(strategy)  # 生成划分子集后的邻接矩阵列表
        # self.source_M 原点阵
    def __str__(self):
        return str(self.A)  # 返回邻接矩阵，加了强转，否则print出错

    def get_edge(self, layout):
        if layout == 'ntu-rgb+d':
            self.num_node = 25
            # self_link = [(i, i) for i in range(self.num_node)]  # 每个节点都与自身相连
            directed_edges = [(i-1, j-1) for i, j in [(1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
                                                      (6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
                                                      (10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
                                                      (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
                                                      (21, 2), (21, 3), (21, 5), (21, 9), (21, 21)
                                                      ]]  # 24个元组，骨骼点的连接关系。各边是有顺序要求的不能换序，后面还要用到边的编号
            # Add self loop for Node 21 (the centre) to avoid singular matrices
            # TODO:数据从哪里来的？表示25个点的简单连接关系，不重复。即论文中的H。
            # neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]  # TODO:是不是nbase-n？不是，骨骼从1开始，列表从0开始，这里生成的是对角矩阵
            self.edge = directed_edges
            # [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24),
            # (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]
            self.center = 21 - 1   # 列表从0开始编号的，骨骼点从1编号的，所以减1
        elif layout == 'openpose':  # 要设计有向，方便demo展示。作者用的openpose处理的kinetics数据集
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]  # 17个元组，骨骼点的连接关系
            self.edge = self_link + neighbor_link  # 度矩阵
            self.center = 1  # TODO: center属性的作用。中心节点是事先规定的，1编号的节点中心节点，且只有一个中心节点
        elif layout == 'ntu_edge':  # TODO:这里新加入了一个candidate?
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'ucla':  # ucla数据集
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    #计算邻接矩阵A
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)  # 有效距离  range(start, stop, step)
        # range(start, stop, step) range(0, the maximal distance between two connected nodes, the spacing between the kernel points)
        adjacency = np.zeros((self.num_node, self.num_node))  # 初始化一个邻接矩阵（方阵）
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
            # self.hop_dis是求距离矩阵的，当距离等于hop时对应位置置为1.这里得到的是一个矩阵，不断填充新的不同阶的邻居  找到所有邻接点
        normalize_adjacency = normalize_digraph(adjacency)  # 矩阵规范化，得到的是规范化的max_hop阶邻接矩阵

        if strategy == 'uniform':  # 同一标签
            A = np.zeros((1, self.num_node, self.num_node))  # 一个矩阵
            A[0] = normalize_adjacency  # 初始化邻接矩阵
            self.A = A

        elif strategy == 'distance':  # todo:n阶相邻标签就是n-1? valid_hop决定了邻接矩阵的个数Ak？label和邻接矩阵有什么关系？同一label具有相同的权重Wk
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))  # 一个矩阵列表, (len(valid_hop)个矩阵
            for i, hop in enumerate(valid_hop):  # 0,0   1,1
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':  # 根据到中心的平均距离分为三类, k = 3  todo:是否已经考虑了distance和spatial的策略融合？
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)  # 加法 各元素对应相加
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        # todo:有个问题，不考虑距离因素划分子集的方法没办法进行多尺度操作。
        elif strategy == 'attribute':  # 根据节点的属性分类 TODO: 还要看A求和问题 ,这里的A不需要相加.没有提相加问题。
            # adjacency得到的是邻域节点
            # self.hop_dis 距离矩阵
            # self.get_adjacency(strategy)  # 生成划分子集后的邻接矩阵列表
            # self.source_M, self.target_M 源点，终点
            A = []
            a_target = np.zeros((self.num_node, self.num_node))
            a_both = np.zeros((self.num_node, self.num_node))
            out_degree = compute_out_degree(self.num_node, self.source_M)
            for i in range(self.num_node):
                if out_degree[i] == 0:
                    a_target[i] = normalize_adjacency[i]  # 这样赋值是可以实现的
                else:
                    a_both[i] = normalize_adjacency[i]
            A.append(a_target)
            A.append(a_both)
            A = np.stack(A)
            self.A = A
        elif strategy == 'dis+attr':  # 根据节点的属性分类 todo：似乎没有办法考虑 一个点在该邻域内是原点还是终点
            # adjacency得到的是邻域节点
            # self.hop_dis 距离矩阵
            # self.get_adjacency(strategy)  # 生成划分子集后的邻接矩阵列表
            # self.source_M, self.target_M 源点，终点
            A = []
            for hop in valid_hop:  # 第一阶
                a_target = np.zeros((self.num_node, self.num_node))
                a_both = np.zeros((self.num_node, self.num_node))
                out_degree = compute_out_degree(self.num_node, self.source_M)
                for i in range(self.num_node):  # todo: 这里遍历不能用 i，j。有的点没边，比如二阶的
                    for j in range(self.num_node):
                        if self.hop_dis[i, j] == hop:
                            # # 1.得到对应边的编号 self.edge.index((i,j))，得到源、终点阵所在的列
                            # edge_id = self.edge.index((i, j))
                            if out_degree[i] == 0: # 既然source_M和target_M列数是边的序列，那么肯定不存在两个都是1的情况
                                a_target[i, j] = normalize_adjacency[i, j]
                            else :
                                a_both[i, j] = normalize_adjacency[i, j]
                A.append(a_target)
                A.append(a_both)
            A = np.stack(A)
            self.A = A
        elif strategy == 'activity':  # Strong, medium, weak
            A = []
            for hop in valid_hop:  # 第一阶
                a_strong = np.zeros((self.num_node, self.num_node))
                a_medium = np.zeros((self.num_node, self.num_node))
                a_weak = np.zeros((self.num_node, self.num_node))
                out_degree = compute_out_degree(self.num_node, self.source_M)
                for i in range(self.num_node):  # todo: 这里遍历不能用 i，j。有的点没边，比如二阶的
                    for j in range(self.num_node):
                        if self.hop_dis[i, j] == hop:
                            # # 1.得到对应边的编号 self.edge.index((i,j))，得到源、终点阵所在的列
                            # edge_id = self.edge.index((i, j))
                            if out_degree[i] == 0: # 既然source_M和target_M列数是边的序列，那么肯定不存在两个都是1的情况
                                a_strong[i, j] = normalize_adjacency[i, j]
                            elif out_degree[i] == 1:
                                a_medium[i, j] = normalize_adjacency[i, j]
                            else:
                                a_weak[i, j] = normalize_adjacency[i, j]
                A.append(a_strong)
                A.append(a_medium)
                A.append(a_weak)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")
        return A, normalize_adjacency

# todo:此函数的返回值hop_dis就是图的1-maxhop阶邻接矩阵的拼接? 返回的是n阶距离矩阵
# todo:有了距离矩阵可以方便地得到任意N接的邻接矩阵
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1   # 只要两点之间有边就是1
        A[i, j] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf  # np.inf 表示一个无穷大的正数
    # np.linalg.matrix_power(A, d)求矩阵A的d幂次方,transfer_mat矩阵是一个将A矩阵依次乘以d并拼接的矩阵
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    # (np.stack(transfer_mat) > 0)矩阵中大于0的返回Ture,小于0的返回False,最终arrive_mat是一个布尔矩阵,大小与transfer_mat一样
    arrive_mat = (np.stack(transfer_mat) > 0)
    # range(start,stop,step) step=-1表示倒着取  1， 0
    for d in range(max_hop, -1, -1):
        # 将arrive_mat[d]矩阵中为True的对应于hop_dis[]位置的数设置为d
        hop_dis[arrive_mat[d]] = d
    return hop_dis  # 元素值只有0~max_hop，inf

# 将矩阵A中的每一列的各个元素分别除以此列元素的和形成新的矩阵
def normalize_digraph(A):
    Dl = np.sum(A, 0)  # 将矩阵A压缩成一行, 每一行对应位置元素累加，相当于求列和.  Dl相当于度矩阵
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD  # 相当于公式（9）中的正则化

# from dgnn  关联矩阵
def normalize_incidence_matrix(im: np.ndarray, full_im: np.ndarray) -> np.ndarray:
    # NOTE:
    # 1. The paper assumes that the Incidence matrix is square,
    #    so that the normalized form A @ (D ** -1) is viable.
    #    However, if the incidence matrix is non-square, then
    #    the above normalization won't work.
    #    For now, move the term (D ** -1) to the front
    # 2. It's not too clear whether the degree matrix of the FULL incidence matrix
    #    should be calculated, or just the target/source IMs.
    #    However, target/source IMs are SINGULAR matrices since not all nodes
    #    have incoming/outgoing edges, but the full IM as described by the paper
    #    is also singular, since ±1 is used for target/source nodes.
    #    For now, we'll stick with adding target/source IMs.
    degree_mat = full_im.sum(-1) * np.eye(len(full_im))
    # Since all nodes should have at least some edge, degree matrix is invertible
    inv_degree_mat = np.linalg.inv(degree_mat)  # 矩阵求逆
    return (inv_degree_mat @ im) + epsilon  # @是Python 3.5之后加入的矩阵乘法运算符 + epsilon to avoid division by zero

# from dgnn
def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    # NOTE: For now, we won't consider all possible edges
    # max_edges = int(special.comb(num_nodes, 2))
    max_edges = len(edges)
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.   # num_nodes, max_edges  (0, 0) = 1  （9, 10）= 1   (10, 11) = 1
        target_graph[target_node, edge_id] = 1.   # num_nodes, max_edges  (13, 0) = 1  (10, 10) = 1  (11, 11) = 1
    full_graph = source_graph + target_graph  # 其实相当于整体的邻接矩阵，有边就是1
    nor_source_graph = normalize_incidence_matrix(source_graph, full_graph)
    nor_target_graph = normalize_incidence_matrix(target_graph, full_graph)
    return source_graph, target_graph, nor_source_graph, nor_target_graph

# from dgnn 这个方法也保存着，后面用来计算谁既是原点又时终点，谁是多终点或者多原点,似乎用矩阵运算更有科学性
def build_digraph_source_list(edges: List[Tuple]) -> np.ndarray:  # ->指定返回值类型
    graph = defaultdict(list)  # 避免键值为空的时候返回KeyError异常，defaultdict()自带默认值
    for source, target in edges:
        graph[source].append(target)
    return graph
# graph {'source':[target1, target2]}
# graph  defaultdict(<class 'list'>, {0: [12, 16], 1: [0], 2: [3], 4: [5], 5: [6], 6: [7],
# 7: [21, 22], 8: [9], 9: [10], 10: [11], 11: [23, 24], 12: [13], 13: [14], 14: [15], 16: [17],
# 17: [18], 18: [19], 20: [1, 2, 4, 8, 20]})
# 因为只有20个非叶子节点

# 新增函数 根据源/终点关联矩阵求各点的出入度
def compute_out_degree(num_node, source_M):  # 本graph所有入度都是1
    out_degree = []
    for i in range(num_node):
        out_degree.append(sum(source_M[i]))
        # in_degree.append(sum(target_M[i]))
    return out_degree


# 新增自写函数,不用加了，根据骨骼结构不会有一个target有两个source的情况
def build_digraph_target_list(edges: List[Tuple]) -> np.ndarray:  # ->指定返回值类型
    pass


if __name__ == '__main__':
    graph = Graph(layout='ntu-rgb+d',  # openpose or ntu-rgb+d  # 改
                  strategy='spatial',  # uniform or distance or spatial  # 改
                  max_hop=3,  # the maximal distance between two connected nodes  # 这里要做成列表  todo:相当于分几类?到中心节点的最大距离
                  dilation=1)
    # print(graph)  # 返回的是A  # max_hop=3, 返回的是9个矩阵,max_hop=2, 返回的是7个矩阵

    (A1, na) = graph.get_adjacency('spatial')
    print("#########na#############")
    print(na)
    print("#########spatial#############")
    print(A1)
    (A2, na)= graph.get_adjacency('attribute')
    print("#########attribute#############")
    print(A2)
    (A3, na)= graph.get_adjacency('dis+attr')
    print("#########dis+attr#############")
    print(A3)
    (A4, na) = graph.get_adjacency('activity')
    print("#########activity#############")
    print(A4)
    degree = compute_out_degree(graph.num_node, graph.source_M)
    print(degree)


    # # print(A)
    # print(np.array(graph.A).shape)
#  todo: 能够发现 随着距离的增大，A的长度会增但增加的矩阵其实是很稀疏的，基本都是0