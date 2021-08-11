import numpy as np
import open3d as o3d
import igraph as ig


def visualizer():
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    cs = coordinate_system()
    vis.add_geometry(cs)
    return vis


def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def to_igraph(graph_dict, unions):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    n_edges = senders.shape[0]
    edge_list = n_edges*[None]
    for i in range(n_edges):
        vi = int(senders[i])
        vj = int(receivers[i])
        edge_list[i] = (vi, vj)
    g = ig.Graph(edges=edge_list)
    g.es["union"] = unions.tolist()
    return g


def partition_vec(ig_graph, n_P, sp_idxs):
    p_vec = np.zeros((n_P, ), dtype=np.uint32)
    unions = ig_graph.es["union"]
    unions = np.array(unions, dtype=np.bool)
    idxs = np.where(unions == False)[0]
    idxs = idxs.astype(np.uint32)
    idxs = idxs.tolist()
    ig_graph.delete_edges(idxs)
    clusters = ig_graph.clusters()
    c = 1
    for i in range(len(clusters)):
        cluster = clusters[i]
        for j in range(len(cluster)):
            sp_i = cluster[j]
            P_idxs = sp_idxs[sp_i]
            p_vec[P_idxs] = c
        c += 1
    return p_vec


def partition_pcd(graph_dict, unions, P, sp_idxs, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    n_P = P.shape[0]
    C = np.zeros((n_P, 3))
    #"""
    g = to_igraph(graph_dict=graph_dict, unions=unions)
    p_vec = partition_vec(ig_graph=g, n_P=n_P, sp_idxs=sp_idxs)
    
    superpoints = np.unique(p_vec)
    n_superpoints = superpoints.shape[0]
    #print(n_superpoints)
    for i in range(n_superpoints):
        superpoint_value = superpoints[i]
        idx = np.where(p_vec == superpoint_value)[0]
        color = colors[i, :]
        #print(len(idx), color)
        C[idx, :] = color / 255
    #"""
    pcd.colors = o3d.utility.Vector3dVector(C)
    return pcd