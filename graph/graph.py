from collections import defaultdict

import numpy as np
from typing import Tuple, List

# num_nodes = 25
epsilon = 1e-6

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:

        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        - activity: Activity Partitioning
        For more information, please refer to our paper (https://ieeexplore.ieee.org/document/10219780).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 layout='ntu-rgb+d',  # openpose or ntu-rgb+d
                 strategy='activity',  # uniform or distance or spatial 
                 max_hop=1,  # the maximal distance between two connected nodes  
                 dilation=1, # controls the spacing between the kernel points  
                 ):  
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)  
        self.source_M, self.target_M, self.nor_source_graph, self.nor_target_graph = build_digraph_incidence_matrix(self.num_node,
                                                                                                                    self.edge)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)


        self.get_adjacency(strategy)  
    def __str__(self):
        return str(self.A)  

    def get_edge(self, layout):
        if layout == 'ntu-rgb+d':
            self.num_node = 25
            directed_edges = [(i-1, j-1) for i, j in [(1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
                                                      (6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
                                                      (10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
                                                      (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
                                                      (21, 2), (21, 3), (21, 5), (21, 9), (21, 21)
                                                      ]] 
           
            self.edge = directed_edges
            # [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24),
            # (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]
            self.center = 21 - 1   
        elif layout == 'openpose':  
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]  
            self.edge = self_link + neighbor_link 
            self.center = 1  
        elif layout == 'ntu_edge':  
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
        elif layout == 'ucla': 
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

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)  
        adjacency = np.zeros((self.num_node, self.num_node)) 
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)  

        if strategy == 'uniform':  
            A = np.zeros((1, self.num_node, self.num_node)) 
            A[0] = normalize_adjacency 
            self.A = A

        elif strategy == 'distance': 
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))  
            for i, hop in enumerate(valid_hop):  # 0,0   1,1
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':  
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
                    A.append(a_root + a_close)  
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        elif strategy == 'activity':  # Strong, medium, weak
            A = []
            for hop in valid_hop: 
                a_strong = np.zeros((self.num_node, self.num_node))
                a_medium = np.zeros((self.num_node, self.num_node))
                a_weak = np.zeros((self.num_node, self.num_node))
                out_degree = compute_out_degree(self.num_node, self.source_M)
                for i in range(self.num_node):  
                    for j in range(self.num_node):
                        if self.hop_dis[i, j] == hop:
                            if out_degree[i] == 0: 
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

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1  
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf  # np.inf 
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis 

def normalize_digraph(A):
    Dl = np.sum(A, 0) 
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
    return DAD  

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
    inv_degree_mat = np.linalg.inv(degree_mat) 
    return (inv_degree_mat @ im) + epsilon 

def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    # NOTE: For now, we won't consider all possible edges
    # max_edges = int(special.comb(num_nodes, 2))
    max_edges = len(edges)
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.   # num_nodes, max_edges  (0, 0) = 1  （9, 10）= 1   (10, 11) = 1
        target_graph[target_node, edge_id] = 1.   # num_nodes, max_edges  (13, 0) = 1  (10, 10) = 1  (11, 11) = 1
    full_graph = source_graph + target_graph  
    nor_source_graph = normalize_incidence_matrix(source_graph, full_graph)
    nor_target_graph = normalize_incidence_matrix(target_graph, full_graph)
    return source_graph, target_graph, nor_source_graph, nor_target_graph

def build_digraph_source_list(edges: List[Tuple]) -> np.ndarray:  
    graph = defaultdict(list)  
    for source, target in edges:
        graph[source].append(target)
    return graph

def compute_out_degree(num_node, source_M): 
    out_degree = []
    for i in range(num_node):
        out_degree.append(sum(source_M[i]))
        # in_degree.append(sum(target_M[i]))
    return out_degree


def build_digraph_target_list(edges: List[Tuple]) -> np.ndarray:  
    pass


