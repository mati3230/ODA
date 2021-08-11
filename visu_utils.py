import numpy as np
import open3d as o3d


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