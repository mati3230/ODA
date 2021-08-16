import numpy as np
import open3d as o3d
import pptk


def pick_sp_points_o3d(pcd):
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


def pick_sp_points_pptk(P, partition=None, initial_partition=None, point_size=-1):
    v = render_pptk_(P=P, partition=partition, initial_partition=initial_partition, point_size=point_size)
    idxs = v.get("selected")
    idxs = np.array(idxs, dtype=np.uint32)
    idxs = np.unique(idxs)
    v.close()
    return idxs


def render_pptk_(P, partition=None, initial_partition=None, point_size=-1):
    if P is None:
        return None
    xyz = P[:, :3]
    rgb = np.array(P[:, 3:], copy=True)
    max_c = np.max(rgb, axis=0)
    max_c = max_c[max_c > 1]
    if max_c.shape[0] > 0:
        #print("Normalize colors.")
        rgb /= 255.
    v = pptk.viewer(xyz)
    if point_size > 0:
        v.set(point_size=point_size)
    colors = P[:, 3:]
    if partition is None and initial_partition is None:
        v.attributes(rgb)
    elif initial_partition is not None and partition is None:
        v.attributes(rgb, initial_partition)
    elif initial_partition is None and partition is not None:
        v.attributes(rgb, partition)
    else:
        v.attributes(rgb, initial_partition, partition)
    print("Press Return to continue.")
    v.wait()
    return v


def render_pptk(P, partition=None, initial_partition=None, point_size=-1):
    v = render_pptk_(P=P, partition=partition, initial_partition=initial_partition, point_size=point_size)
    if v is None:
        return
    v.close()


def render_o3d(x):
    if type(x) == list:
        x.append(coordinate_system())
        o3d.visualization.draw_geometries(x)
        x.pop(-1)
    else:
        o3d.visualization.draw_geometries([x, coordinate_system()])


def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set