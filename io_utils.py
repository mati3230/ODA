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
    """ Load colors in order to visualize.
    
    Returns
    -------
    np.ndarray
        n X 3 array (8 bit).
    """
    print("Load colors...")
    data = np.load("colors.npz")
    colors = data["colors"]
    print("Done")
    return colors


def load_unions(fdir, graph_dict, filename=""):
    """ Load the link union decisions of the agent.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    graph_dict : dict
        Dictionary where the superpoint graph is stored.
    filename : str
        Name of the file.
    
    Returns
    -------
    np.ndarray, dict
        Binary array where the link predictions are stored. 
        Dictionary where the superpoint graph is stored.

    """
    print("Load unions...")
    file = "{0}/unions_{1}.h5".format(fdir, filename)
    if not file_exists(filepath=file):
        return None
    hf = h5py.File(file, "r")
    unions = np.array(hf["unions"], copy=True)
    senders = np.array(hf["senders"], copy=True)
    receivers = np.array(hf["receivers"], copy=True)
    hf.close()
    graph_dict["senders"] = senders
    graph_dict["receivers"] = receivers
    print("Done")
    return unions, graph_dict


def save_unions(fdir, unions, graph_dict, filename=""):
    """ Save the link union decisions of the agent.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    unions : np.ndarray
        Binary array where the link predictions are stored.
    graph_dict : dict
        Dictionary where the superpoint graph is stored.
    filename : str
        Name of the file.
    """
    print("Save unions")
    mkdir(fdir)
    hf = h5py.File("{0}/unions_{1}.h5".format(fdir, filename), "w")
    hf.create_dataset("unions", data=unions)
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    hf.create_dataset("senders", data=senders)
    hf.create_dataset("receivers", data=receivers)
    hf.close()
    print("Done")


def save_probs(fdir, P, graph_dict, sp_idxs, probs, initial_db, save_init=False, filename="", half="", stris=None):
    """ Save the link union decisions of the agent.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    P : np.ndarray
        The point cloud.
    graph_dict : dict
        Dictionary where the superpoint graph is stored.
    sp_idxs : list[np.ndarray]
        List of the point indices for each superpoint.
    probs : np.ndarray
        Probabilities of the link predictions which are used to create the unions.
    initial_db : float
        TODO
    save_init : bool
        Should the initial partition be stored?
    filename : str
        Name of the file.
    half : str
        TODO
    """
    print("Save probs")
    if save_init:
        save_init_graph(
            fdir=fdir,
            P=P,
            graph_dict=graph_dict,
            sp_idxs=sp_idxs,
            half=half,
            stris=stris)
    mkdir(fdir)
    hf = h5py.File("{0}/probs_{1}.h5".format(fdir, filename), "w")
    hf.create_dataset("probs", data=probs)
    hf.create_dataset("initial_db", data=(initial_db, ))
    hf.close()
    print("Done")


def load_probs(fdir, filename=""):
    """ Load the probabilities of the link predictions.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    filename : str
        Name of the file.
    
    Returns
    -------
    np.ndarray, dict, list[np.ndarray], np.ndarray, np.ndarray
        The point cloud.
        Dictionary where the superpoint graph is stored.
        List of the point indices for each superpoint.
        Probabilities of the link predictions which are used to create the unions.
        Binary array where the link predictions are stored. 

    """
    print("Load probs...")
    P, graph_dict, sp_idxs = load_init_graph(fdir=fdir, filename=filename)
    if P is None:
        return None, None, None, None, None, None
    print("Point cloud size: {0} points".format(P.shape[0]))

    file = "{0}/probs_{1}.h5".format(fdir, filename)
    if not file_exists(filepath=file):
        return None, None, None, None, None, None
    hf = h5py.File(file, "r")
    probs = np.array(hf["probs"], copy=True)
    initial_db = float(hf["initial_db"][0])
    unions = np.zeros((probs.shape[0], ), dtype=np.bool)
    unions[probs > initial_db] = True
    hf.close()
    print("Done")
    return P, graph_dict, sp_idxs, probs, unions


def load_probs_mesh(fdir, filename=""):
    """ Load the probabilities of the link predictions.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    filename : str
        Name of the file.
    
    Returns
    -------
    np.ndarray, dict, list[np.ndarray], np.ndarray, np.ndarray
        The point cloud.
        Dictionary where the superpoint graph is stored.
        List of the point indices for each superpoint.
        Probabilities of the link predictions which are used to create the unions.
        Binary array where the link predictions are stored. 

    """
    print("Load probs...")
    mesh, graph_dict, sp_idxs, stris = load_init_graph_mesh(fdir=fdir, filename=filename)
    if mesh is None:
        return None, None, None, None, None, None
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    print("Mesh size: {0} vertices, {1} triangles".format(vertices.shape[0], triangles.shape[0]))

    file = "{0}/probs_{1}.h5".format(fdir, filename)
    if not file_exists(filepath=file):
        return None, None, None, None, None, None
    hf = h5py.File(file, "r")
    probs = np.array(hf["probs"], copy=True)
    initial_db = float(hf["initial_db"][0])
    unions = np.zeros((probs.shape[0], ), dtype=np.bool)
    unions[probs > initial_db] = True
    hf.close()
    print("Done")
    return mesh, graph_dict, sp_idxs, probs, unions, stris


"""
save_init_graph(
    fdir=args.g_dir,
    P=mesh, graph_dict=graph_dict,
    sp_idxs=sp_idxs,
    filename=args.g_filename,
    stris=stris)
"""
def save_init_graph(fdir, P, graph_dict, sp_idxs, filename="", half="", stris=None):
    """ Save the initial partition.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    P : np.ndarray
        The point cloud.
    graph_dict : dict
        Dictionary where the superpoint graph is stored.
    sp_idxs : list[np.ndarray]
        List of the point indices for each superpoint.
    filename : str
        Name of the file.
    half : str
        TODO
    """
    print("Save initial graph")
    mkdir(fdir)
    hf = h5py.File("{0}/init_graph{1}_{2}.h5".format(fdir, half, filename), "w")
    n_sps = len(sp_idxs)
    hf.create_dataset("n_sps", data=(n_sps, ))
    if stris is None:
        hf.create_dataset("P", data=P)
        for i in range(n_sps):
            hf.create_dataset(str(i), data=sp_idxs[i])
    else:
        vertices = np.asarray(P.vertices)
        vertex_colors = np.asarray(P.vertex_colors)
        triangles = np.asarray(P.triangles)
        hf.create_dataset("vertices", data=vertices)
        hf.create_dataset("vertex_colors", data=vertex_colors)
        hf.create_dataset("triangles", data=triangles)
        for i in range(n_sps):
            hf.create_dataset(str(i), data=sp_idxs[i])
            hf.create_dataset("t" + str(i), data=stris[i])

    nodes = graph_dict["nodes"]
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    hf.create_dataset("nodes", data=nodes)
    hf.create_dataset("senders", data=senders)
    hf.create_dataset("receivers", data=receivers)
    hf.close()
    print("Done")


def load_init_graph_mesh(fdir, filename="", half=""):
    """ Load the probabilities of the link predictions.

    Parameters
    ----------
    fdir : str
        Relative or absolute directory of the file.
    filename : str
        Name of the file.
    half : str
        TODO
    
    Returns
    -------
    np.ndarray, dict, list[np.ndarray], np.ndarray, np.ndarray
        The point cloud.
        Dictionary where the superpoint graph is stored.
        List of the point indices for each superpoint.

    """
    print("Load initial graph...")
    file = "{0}/init_graph{1}_{2}.h5".format(fdir, half, filename)
    if not file_exists(filepath=file):
        #print(file)
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
    vertices = np.array(hf["vertices"], copy=True)
    vertex_colors = np.array(hf["vertex_colors"], copy=True)
    triangles = np.array(hf["triangles"], copy=True)
    print("Mesh size: {0} vertices, {1} triangles".format(vertices.shape[0], triangles.shape[0]))
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles)
        )
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    n_sps = int(hf["n_sps"][0])
    sp_idxs = n_sps * [None]
    stris = n_sps * [None]
    for i in range(n_sps):
        idxs = np.array(hf[str(i)], copy=True)
        sp_idxs[i] = idxs
        stris[i] = np.array(hf["t"+str(i)], copy=True)
    sp_idxs = np.array(sp_idxs, dtype = "object")
    hf.close()
    print("Done")
    return mesh, graph_dict, sp_idxs, stris


def check_color_value(c):
    """ Check if there is an invalid colour value.
    """
    if c > 255 or c < 0:
        raise Exception("Invalid color value: {0}".format(c))


def load_partition(file):
    if file.endswith("npz"):
        data = np.load(file, allow_pickle=True)
        partition = data["partition"]
    if file.endswith("txt"):
        partition = np.loadtxt(file)
    if file.endswith("h5"):
        hf = h5py.File(file, "r")
        partition = np.array(hf["partition"], copy=True)
        hf.close()
    print("Partition loaded")
    return partition


def save_partition(partition, fdir, fname):
    print("Save partition")
    hf = h5py.File("{0}/partition_{1}.h5".format(fdir, fname), "w")
    hf.create_dataset("partition", data=partition)
    hf.close()
    print("Done")


def save_partitions(partitions, fdir, fname):
    print("Save partitions")
    hf = h5py.File("{0}/partitions_{1}.h5".format(fdir, fname), "w")
    for pname, partition in partitions:
        hf.create_dataset(pname, data=partition)
    hf.close()
    print("Done")


def load_cloud(file, r=255, g=0, b=0, p=1):
    """ Load a point cloud.

    Parameters
    ----------
    file : str
        Name of the file.
    r : uint8
        Fallback red color.
    g : uint8
        Fallback green color.
    b : uint8
        Fallback blue color.
    p : float
        Sampling rate lower than 1 can be used to uniformly downsample the point cloud.
    
    Returns
    -------
    np.ndarray
        The Nx6 point cloud.

    """
    print("Load point cloud...")
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
    elif file.endswith("xyz"):
        P = from_txt(file=file, r=r, g=g, b=b, p=p)
    elif file.endswith("xyzrgb"):
        P = from_txt(file=file, r=r, g=g, b=b, p=p)
    else:
        raise Exception("Unknwon file type: {0}".format(file))
    print("Point cloud size: {0} points".format(P.shape[0]))
    print("Done")
    return P


def load_proc_cloud(fdir, fname):
    hf = h5py.File("{0}/P_{1}.h5".format(fdir, fname), "r")
    P = np.array(hf["P"], copy=True)
    print("Point cloud size: {0} points".format(P.shape[0]))
    hf.close()
    return P


def load_proc_mesh(fdir, fname):
    hf = h5py.File("{0}/mesh_{1}.h5".format(fdir, fname), "r")
    vertices = np.array(hf["vertices"], copy=True)
    vertex_colors = np.array(hf["vertex_colors"], copy=True)
    triangles = np.array(hf["triangles"], copy=True)
    print("Mesh size: {0} vertices, {1} triangles".format(vertices.shape[0], triangles.shape[0]))
    hf.close()
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices), 
        triangles=o3d.utility.Vector3iVector(triangles)
        )
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def load_nn_file(fdir, fname, a):
    print("Load nearest neighbours")
    hf = h5py.File("{0}/nn_{1}_{2}.h5".format(fdir, fname, a), "r")

    distances = np.array(hf["distances"], copy=True)
    edge_weight = np.array(hf["edge_weight"], copy=True)
    source = np.array(hf["source"], copy=True)
    target = np.array(hf["target"], copy=True)

    d_mesh = {
        "distances": distances,
        "edge_weight": edge_weight,
        "source": source,
        "target": target
    }
    print("Done")
    return d_mesh


def save_nn_file(fdir, fname, a ,d_mesh):
    print("Save nearest neighbours")
    hf = h5py.File("{0}/nn_{1}_{2}.h5".format(fdir, fname, a), "w")
    hf.create_dataset("source", data=d_mesh["source"])
    hf.create_dataset("target", data=d_mesh["target"])
    hf.create_dataset("distances", data=d_mesh["distances"])
    hf.create_dataset("edge_weight", data=d_mesh["edge_weight"])
    hf.close()
    print("Done")


def save_cloud(P, fdir, fname):
    if type(P) == o3d.geometry.TriangleMesh:
        print("Save mesh")
        hf = h5py.File("{0}/mesh_{1}.h5".format(fdir, fname), "w")
        vertices = np.asarray(P.vertices)
        vertex_colors = np.asarray(P.vertex_colors)
        triangles = np.asarray(P.triangles)
        hf.create_dataset("vertices", data=vertices)
        hf.create_dataset("vertex_colors", data=vertex_colors)
        hf.create_dataset("triangles", data=triangles)
        hf.close()
        print("Done")
    else:
        print("Save cloud")
        hf = h5py.File("{0}/P_{1}.h5".format(fdir, fname), "w")
        hf.create_dataset("P", data=P)
        hf.close()
        print("Done")


def cloud(P, r=255, g=0, b=0, p=1):
    colors_available = P.shape[1] == 6
    P = colorize(P=P, colors_available=colors_available, r=r, g=g, b=b)
    print("Original Point Cloud Size: {0}".format(P.shape[0]))
    P = subsample(P=P, p=p)
    return P


def from_txt(file, r=255, g=0, b=0, p=1, delimiter=" "):
    P = np.loadtxt(file, delimiter=delimiter)
    P = cloud(P=P, r=r, g=g, b=b, p=p)
    return P


def from_npz(file, r=255, g=0, b=0, p=1):
    data = np.load(file, allow_pickle=True)
    try:
        P = data["P"]
    except KeyError as  ke:
        xyz = data["xyz"]
        rgb = data["rgb"]
        P = np.hstack((xyz, rgb))
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
    print("Point cloud size before sampling: {0}".format(P.shape[0]))
    size = math.floor(p * P.shape[0])
    idxs = np.arange(P.shape[0])
    idxs = np.random.choice(a=idxs, size=size, replace=False)
    P = P[idxs]
    print("Point cloud size after sampling: {0}".format(P.shape[0]))
    return P


def save_mesh(mesh, fdir, filename, o_id, ending="glb"):
    mkdir(fdir)
    file = "{0}/mesh_{1}_{2}.{3}".format(fdir, filename, o_id, ending)
    print("Save mesh: {0}".format(file))
    o3d.io.write_triangle_mesh(file, mesh)
    print("Done")

def save_meshes(meshes, fdir, filename="", ending="glb"):
    print("Save meshes")
    mkdir(fdir)
    for i in range(len(meshes)):
        mesh = meshes[i]
        save_mesh(mesh=mesh, fdir=fdir, filename=filename, o_id=i, ending=ending)


def load_mesh(file):
    mesh = o3d.io.read_triangle_mesh(file)
    return mesh


def write_csv(filedir, filename, csv):
    if filedir[-1] != "/":
        filedir += "/"
    if not filename.endswith(".csv"):
        filename += ".csv"
    out_filename = filedir + filename
    file = open(out_filename, "w")
    file.write(csv)
    file.close()