import numpy as np
import open3d as o3d
import math
import os
import h5py


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    """Check if a file exists.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to a file.

    Returns
    -------
    boolean
        True if the file exists.

    """
    return os.path.isfile(filepath)


def load_colors():
    data = np.load("colors.npz")
    colors = data["colors"]
    return colors


def load_unions(fdir, graph_dict):
    file = fdir + "/unions.h5"
    if not file_exists(filepath=file):
        return None
    hf = h5py.File(file, "r")
    unions = np.array(hf["unions"], copy=True)
    senders = np.array(hf["senders"], copy=True)
    receivers = np.array(hf["receivers"], copy=True)
    hf.close()
    graph_dict["senders"] = senders
    graph_dict["receivers"] = receivers
    return unions, graph_dict


def save_unions(fdir, unions, graph_dict):
    mkdir(fdir)
    hf = h5py.File(fdir + "/unions.h5", "w")
    hf.create_dataset("unions", data=unions)
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    hf.create_dataset("senders", data=senders)
    hf.create_dataset("receivers", data=receivers)
    hf.close()


def save_probs(fdir, P, graph_dict, sp_idxs, probs, initial_db, save_init=False):
    if save_init:
        save_init_graph(fdir=fdir, P=P, graph_dict=graph_dict, sp_idxs=sp_idxs)
    mkdir(fdir)
    hf = h5py.File(fdir + "/probs.h5", "w")
    hf.create_dataset("probs", data=probs)
    hf.create_dataset("initial_db", data=(initial_db, ))
    hf.close()

def load_probs(fdir):
    P, graph_dict, sp_idxs = load_init_graph(fdir=fdir)

    file = fdir + "/probs.h5"
    if not file_exists(filepath=file):
        return None, None, None, None, None, None
    hf = h5py.File(file, "r")
    probs = np.array(hf["probs"], copy=True)
    initial_db = float(hf["initial_db"][0])
    unions = np.zeros((probs.shape[0], ), dtype=np.bool)
    unions[probs > initial_db] = True
    hf.close()
    
    return P, graph_dict, sp_idxs, probs, unions

def save_init_graph(fdir, P, graph_dict, sp_idxs):
    mkdir(fdir)
    hf = h5py.File(fdir + "/init_graph.h5", "w")
    hf.create_dataset("P", data=P)
    n_sps = len(sp_idxs)
    hf.create_dataset("n_sps", data=(n_sps, ))
    for i in range(n_sps):
        hf.create_dataset(str(i), data=sp_idxs[i])
    nodes = graph_dict["nodes"]
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    hf.create_dataset("nodes", data=nodes)
    hf.create_dataset("senders", data=senders)
    hf.create_dataset("receivers", data=receivers)
    hf.close()


def load_init_graph(fdir):
    file = fdir + "/init_graph.h5"
    if not file_exists(filepath=file):
        return None, None, None
    hf = h5py.File(file, "r")
    nodes = np.array(hf["nodes"], copy=True)
    senders = np.array(hf["senders"], copy=True)
    receivers = np.array(hf["receivers"], copy=True)
    graph_dict = {
        "nodes": nodes,
        "senders": senders,
        "receivers": receivers,
        "edges": None,
        "globals": None
    }
    P = np.array(hf["P"], copy=True)
    n_sps = int(hf["n_sps"][0])
    sp_idxs = n_sps * [None]
    for i in range(n_sps):
        idxs = np.array(hf[str(i)], copy=True)
        sp_idxs[i] = idxs
    sp_idxs = np.array(sp_idxs, dtype = "object")
    hf.close()
    return P, graph_dict, sp_idxs


def check_color_value(c):
    if c > 255 or c < 0:
        raise Exception("Invalid color value: {0}".format(c))


def load_cloud(file, r=255, g=0, b=0, p=1):
    if not os.path.exists(file):
        raise Exception("File not found: {0}".format(file))
    check_color_value(r)
    check_color_value(g)
    check_color_value(b)
    if file.endswith("npz"):
        P = from_npz(file=file, r=r, g=g, b=b, p=p)
    elif file.endswith("pcd"):
        P = from_pcd(file=file, r=r, g=g, b=b, p=p)
    elif file.endswith("txt"):
        P = from_txt(file=file, r=r, g=g, b=b, p=p)
    else:
        raise Exception("Unknwon file type: {0}".format(file))
    return P


def cloud(P, r=255, g=0, b=0, p=1):
    colors_available = P.shape[1] == 6
    P = colorize(P=P, colors_available=colors_available, r=r, g=g, b=b)
    P = subsample(P=P, p=p)
    return P


def from_txt(file, r=255, g=0, b=0, p=1, delimiter=" "):
    P = np.loadtxt(file, delimiter=delimiter)
    P = cloud(P=P, r=r, g=g, b=b, p=p)
    return P


def from_npz(file, r=255, g=0, b=0, p=1):
    data = np.load(file, allow_pickle=True)
    P = data["P"]
    P = cloud(P=P, r=r, g=g, b=b, p=p)
    return P


def from_pcd(file, r=255, g=0, b=0, p=1):
    pcd = o3d.io.read_point_cloud(file)
    P = np.asarray(pcd.points)
    if pcd.colors is not None:
        C = np.asarray(pcd.colors)
        P = np.hstack((P, C))
    P = cloud(P=P, r=r, g=g, b=b, p=p)
    return P


def colorize(P, colors_available, r=255, g=0, b=0):
    if not colors_available:
        C = np.zeros((P.shape[0], 3))
        C[:,0] = r
        C[:,1] = g
        C[:,2] = b
        P = np.hstack((P, C))
    return P


def subsample(P, p):
    if p > 1 or p <= 0:
        raise Exception("Cannot sample {0}% of the data.".format(p))
    if p == 1:
        return P
    size = math.floor(p * P.shape[0])
    idxs = np.arange(P.shape[0])
    idxs = np.random.choice(a=idxs, size=size, replace=False)
    P = P[idxs]
    return P


def save_meshes(meshes, fdir):
    mkdir(fdir)
    for i in range(len(meshes)):
        mesh = meshes[i]
        o3d.io.write_triangle_mesh(fdir + "/mesh_" + str(i) + ".obj", mesh)