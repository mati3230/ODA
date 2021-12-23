import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from ai_utils import search_bfs
from visu_utils import pick_sp_points_o3d, render_o3d


def interp(d, c_far=np.array([0, 1, 0]), c_close=np.array([1, 0, 0])):
    d_ = d / np.max(d)
    d_ = d_.reshape(d_.shape[0], 1)
    c_close = c_close.reshape(1, c_close.shape[0])
    c_far = c_far.reshape(1, c_far.shape[0])
    C = np.matmul(d_, c_close) + np.matmul((1-d_), c_far)
    return C 


def colorize(mesh, C, p_idx, nn_distances, nn_targets):
    C_ = np.array(C, copy=True)
    C_[p_idx] = np.array([0, 1, 0])
    C_interp = interp(d=nn_distances)
    C_[nn_targets] = C_interp
    mesh.vertex_colors = o3d.utility.Vector3dVector(C_)


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/home/mati3230/Projects/Datasets/sn000000.ply", type=str, help="Path to a mesh")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.file)

    mesh_vertices = np.asarray(mesh.vertices)
    
    tree = KDTree(data=mesh_vertices)
    
    mesh.compute_adjacency_list()
    adj_list = mesh.adjacency_list

    # extract the neighbourhood of each point
    neighbours = []
    v_idxs = []
    vi = 0
    for al in adj_list:
        v_idxs.extend(len(al) * [vi])
        neighbours.extend(list(al))
        vi += 1
    # edges as 2Xn array
    edges = np.zeros((2, len(neighbours)), dtype=np.uint32)
    edges[0, :] = v_idxs
    edges[1, :] = neighbours
    edges = np.unique(edges, axis=1)
    
    # sort the source vertices
    sortation = np.argsort(edges[0])
    edges = edges[:, sortation]
    
    # edges in source, target layout
    source = edges[0]
    target = edges[1]

    # distances of the edges from a vertex v_i to v_j 
    distances = np.sqrt(np.sum((mesh_vertices[source] - mesh_vertices[target])**2, axis=1))
    uni_verts, direct_neigh_idxs, n_edges = np.unique(edges[0, :], return_index=True, return_counts=True)

    C = np.array(np.asarray(mesh.vertex_colors), copy=True)

    while True:
        try:
            k = int(input("k>=2 [-1: exit]: "))
            if k == -1:
                return
            if k <= 1:
                raise Exception("Please choose k >= 2")
        except Exception as e:
            continue
        p_idxs = pick_sp_points_o3d(pcd=mesh, is_mesh=True)
        try:
            p_idx = p_idxs[0]
        except Exception as e:
            continue

        nns = search_bfs(vi=p_idx, edges=edges, distances=distances, 
            direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges, k=k)
        nn_targets = nns[1, :].astype(np.uint32)
        nn_distances = nns[2, :]

        colorize(mesh=mesh, C=C, p_idx=p_idx, nn_distances=nn_distances, nn_targets=nn_targets)
        render_o3d(x=mesh, w_co=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)

        v = mesh_vertices[p_idx]
        nn_distances, nn_targets = tree.query(x=v, k=k+1)
        nn_targets = nn_targets[1:].astype(np.uint32)
        nn_distances = nn_distances[1:]

        colorize(mesh=mesh, C=C, p_idx=p_idx, nn_distances=nn_distances, nn_targets=nn_targets)
        render_o3d(x=mesh, w_co=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)

if __name__ == "__main__":
    main()