import numpy as np
import open3d as o3d

from sp_utils import comp_list


def pick_sp_points(pcd):
    print("")
    print("1) Pick superpoints that should by unified by using [shift + left click]")
    print("Press [shift + right click] to undo a selection")
    print("2) After picking the superpoints, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def render(x):
    if x is None:
        return
    pcd = x
    if type(x) == np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(x[:, 3:] / 255.)
    elif type(x) == list:
        o3d.visualization.draw_geometries(x)
        return        
    o3d.visualization.draw_geometries([pcd])


def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def initial_partition_pcd(P, sp_idxs, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    n_P = P.shape[0]
    C = np.zeros((n_P, 3))
    n_sps = len(sp_idxs)
    for i in range(n_sps):
        idxs = sp_idxs[i]
        color = colors[i, :]
        #print(len(idx), color)
        C[idxs, :] = color / 255
    pcd.colors = o3d.utility.Vector3dVector(C)
    return pcd


def partition_pcd(graph_dict, unions, P, sp_idxs, colors, col_d=0.25):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    n_P = P.shape[0]
    C = np.zeros((n_P, 3))

    c_list = comp_list(graph_dict=graph_dict, unions=unions, n_P=n_P, sp_idxs=sp_idxs)

    for i in range(len(c_list)):
        comp = c_list[i]
        color = colors[i, :]
        n_sp_comp = len(comp)
        if col_d > 0:
            c_factors = (np.arange(n_sp_comp) + 1) / n_sp_comp
            c_factors *= col_d
            c_factors += (1-col_d)
        else:
            c_factors = np.ones((n_sp_comp, ))
        for j in range(n_sp_comp):
            P_idxs = comp[j][1]
            C[P_idxs, :] = c_factors[j] * color / 255.

    pcd.colors = o3d.utility.Vector3dVector(C)
    return pcd