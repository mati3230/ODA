import numpy as np
import open3d as o3d
import pptk
import matplotlib.pyplot as plt
from tqdm import tqdm


def pick_sp_points_o3d(pcd, is_mesh=False):
    print("")
    print("1) Pick superpoints that should by unified by using [shift + left click]")
    print("Press [shift + right click] to undo a selection")
    print("2) After picking the superpoints, press 'Q' to close the window")
    if is_mesh:
        vis = o3d.visualization.VisualizerWithVertexSelection()
    else:
        vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    pp = vis.get_picked_points()
    picked_points = len(pp)*[None]
    for i in range(len(pp)):
        picked_points[i] = pp[i].index
    return picked_points

def pick_sp_points_pptk(P, partition=None, initial_partition=None, point_size=-1, v=None, colors=None):
    v = render_pptk_(P=P, partition=partition, initial_partition=initial_partition, point_size=point_size, v=v, colors=colors)
    idxs = v.get("selected")
    idxs = np.array(idxs, dtype=np.uint32)
    idxs = np.unique(idxs)
    return idxs, v


def get_perspective(viewer):
    x, y, z = viewer.get('eye')
    phi = viewer.get('phi')
    theta = viewer.get('theta')
    r = viewer.get('r')
    return [x, y, z, phi, theta, r]


def set_perspective(viewer, p):
    if p is None:
        return
    viewer.set(lookat=p[:3], phi=p[3], theta=p[4], r=p[5])


def render_pptk_(P, partition=None, initial_partition=None, point_size=-1, v=None, colors=None):
    if P is None:
        return None
    xyz = P[:, :3]
    rgb = np.array(P[:, 3:], copy=True)
    max_c = np.max(rgb, axis=0)
    max_c = max_c[max_c > 1]
    if max_c.shape[0] > 0:
        #print("Normalize colors.")
        rgb /= 255.
    if v is None:
        v = pptk.viewer(xyz)
    else:
        #persp = get_perspective(viewer=v)
        v.clear()
        v.load(xyz)
    if colors is not None:
        v.color_map(c=colors)
    if point_size > 0:
        v.set(point_size=point_size)
    if partition is None and initial_partition is None:
        v.attributes(rgb)
    elif initial_partition is not None and partition is None:
        v.attributes(rgb, initial_partition)
    elif initial_partition is None and partition is not None:
        v.attributes(rgb, partition)
    else:
        v.attributes(rgb, initial_partition, partition)
    print("Press Return in the 3D windows to continue.")
    #set_perspective(viewer=v, p=persp)
    v.wait()
    return v


def render_pptk_feat(P, point_size=-1, v=None, feats=None):
    if P is None:
        return None
    if feats is None:
        return None
    xyz = P[:, :3]
    rgb = np.array(P[:, 3:], copy=True)
    max_c = np.max(rgb, axis=0)
    max_c = max_c[max_c > 1]
    if max_c.shape[0] > 0:
        #print("Normalize colors.")
        rgb /= 255.
    if v is None:
        v = pptk.viewer(xyz)
    else:
        #persp = get_perspective(viewer=v)
        v.clear()
        v.load(xyz)
    if point_size > 0:
        v.set(point_size=point_size)
    #feats.append(rgb)
    v.attributes(rgb, *feats)
    print("Press Return in the 3D windows to continue.")
    #set_perspective(viewer=v, p=persp)
    v.wait()
    return v


def render_pptk(P, partition=None, initial_partition=None, point_size=-1, v=None, colors=None):
    v = render_pptk_(P=P, partition=partition, initial_partition=initial_partition, point_size=point_size, v=v, colors=colors)
    return v


def render_partition_vec_o3d(mesh, partition, colors):
    vertices = np.asarray(mesh.vertices)
    n_vert = vertices.shape[0]
    rgb = np.zeros((n_vert, 3), dtype=np.float32)
    uni_p = np.unique(partition)
    for i in range(len(uni_p)):
        sp_v = uni_p[i]
        idxs = np.where(partition == sp_v)[0]
        color = colors[i]
        rgb[idxs] = color
    pmesh = o3d.geometry.TriangleMesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles)
    pmesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    render_o3d(pmesh)
    return pmesh

def render_partition_o3d(mesh, sp_idxs, colors, w_co=False):
    vertices = np.asarray(mesh.vertices)
    n_vert = vertices.shape[0]
    rgb = np.zeros((n_vert, 3), dtype=np.float32)
    for i in range(len(sp_idxs)):
        sp = sp_idxs[i]
        color = colors[i]
        rgb[sp] = color
    pmesh = o3d.geometry.TriangleMesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles)
    pmesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    render_o3d(pmesh, w_co=w_co)
    return pmesh


def render_o3d(x, w_co=False):
    if type(x) == list:
        if w_co:
            x.append(coordinate_system())
        o3d.visualization.draw_geometries(x)
        if w_co:
            x.pop(-1)
    else:
        if w_co:
            #cols = np.asarray(x.vertex_colors)
            #cols[:, :] = 0.5
            #cols[271, 0] = 1
            #verts = np.asarray(x.vertices)
            #x.vertex_colors = o3d.utility.Vector3dVector(cols)
            #mesh = o3d.geometry.TriangleMesh()
            #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=5)
            #sphere.translate(verts[271])
            #sphere.paint_uniform_color(np.array([1,0,0]))
            o3d.visualization.draw_geometries([x, coordinate_system()])
            #o3d.visualization.draw_geometries([x, coordinate_system(), sphere])
        else:
            o3d.visualization.draw_geometries([x])


def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def visu_dec_bs(graph_dict, P, sp_idxs, probs, partition_func):
    dbs = np.arange(start=0, stop=1, step=0.01)
    n_parts = np.ones((dbs.shape[0], ), dtype=np.uint32)
    is_mesh = False
    if type(P) == o3d.geometry.TriangleMesh:
        P = np.asarray(P.vertices)
    for i in tqdm(range(dbs.shape[0]), desc="Partition:"):
        db = dbs[i]
        unions = np.zeros((probs.shape[0], ), dtype=np.bool)
        unions[probs > db] = True
        part = partition_func(
            graph_dict=graph_dict,
            unions=unions,
            P=P,
            sp_idxs=sp_idxs)
        uni_part = np.unique(part)
        n_parts[i] = uni_part.shape[0]
    plt.plot(dbs, n_parts)
    plt.title("Number of Superpoints")
    plt.xlabel("Decision Boundary")
    plt.ylabel("Number of Superpoints")
    plt.axis([0, 1, 1, int(np.max(n_parts))])
    plt.show()
    