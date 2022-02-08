# cut pursuit library (see https://github.com/loicland/superpoint_graph/tree/ssp%2Bspg/partition)
try:
    import cp_ext as libcp
    import ply_c_ext as libply_c
except:
    import libcp
    import libply_c

import numpy as np
import numpy.matlib
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay, KDTree
from graph_nets import utils_tf
from tqdm import tqdm
from multiprocessing import Pool
import time
import sys
from multiprocessing.pool import ThreadPool

from network import FFGraphNet

from io_utils import load_nn_file, save_nn_file


def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0, verbose=True):
    """ This function is developed by Landrieu et al. (see https://github.com/loicland/superpoint_graph). 
    Basically, it computes a nearest neighbour graph.  

    Parameters
    ----------
    xyz : np.ndarray
        Points of the point cloud.
    k_nn1 : int
        The k nearest neighbours. k_nn1 must be smaller than k_nn2. This parameter is used for the returned neighbourhood.
    k_nn2 : int
        TODO this parameter can be eliminated? k_nn1 can be used instead????
    voronoi : float
        Distance threshold > 0 which is used in the voronoi graph construction. 

    Returns
    -------
    graph : dict
        Dictionary consisting of the keys: 
        - is_nn: This is always True
        - source: graph source points of an edge
        - target: graph target points of an edge
        - distances: distance of each edge
    target2: np.ndarray
        Shape is (n_ver * k_nn1, ) and is the same as graph["target"] when voronoi = 0.0. 
        This does not hold if voronoi method is used.
    """
    if verbose:
        print("Compute Graph NN")
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    #assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    # get the k nearest neighbours of every point in the point cloud.
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    # row wise flattening with shape: (n_ver * k_nn1, )
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
        # --- We do not use the voronoi functionality yet ---
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:,0],tri.vertices[:,0], \
              tri.vertices[:,0], tri.vertices[:,1], tri.vertices[:,1], tri.vertices[:,2])).astype('uint64')
        graph["target"]= np.hstack((tri.vertices[:,1],tri.vertices[:,2], \
              tri.vertices[:,3], tri.vertices[:,2], tri.vertices[:,3], tri.vertices[:,3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"],:] - xyz[graph["target"],:])**2).sum(1)
        keep_edges = graph["distances"]<voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]
        
        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] =  np.hstack((graph["target"],np.transpose(neighbors.flatten(order='C')).astype('uint32')))
        
        edg_id = graph["source"] + n_ver * graph["target"]
        
        dump, unique_edges = np.unique(edg_id, return_index = True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]
       
        graph["distances"] = graph["distances"][keep_edges]
    else:
        # matrix n_ver X k_nn1
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]

        """
        creating a source target neighbourhood representation for each edge.
        source and target are vectors of size (n_ver * k_nn1, ). 
        """
        
        """
        with n_ver = 100 and k_nn1 = 15
        >>> M = np.matlib.repmat(range(0, n_ver), k_nn1, 1)
        >>> M
        array([[ 0,  1,  2, ..., 97, 98, 99],
               [ 0,  1,  2, ..., 97, 98, 99],
               [ 0,  1,  2, ..., 97, 98, 99],
               ...,
               [ 0,  1,  2, ..., 97, 98, 99],
               [ 0,  1,  2, ..., 97, 98, 99],
               [ 0,  1,  2, ..., 97, 98, 99]])
        Flatten happens column wise with shape: (n_ver * k_nn1, )
        >>> M.flatten(order="F")
        array([ 0,  0,  0, ..., 99, 99, 99])
        """
        graph["source"] = np.matlib.repmat(range(0, n_ver), k_nn1, 1).flatten(order='F').astype('uint32')
        # neighbors.flatten(order='C') -> row wise (default) flatten with shape: (n_ver * k_nn1, )
        # TODO taking the transpose is eventually unnecessary? 
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        # distances between the i-th point and a neighbour point
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    if verbose:
        print("Done")
    return graph, target2


def superpoint_graph(xyz, rgb, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.1, d_se_max=0, 
        verbose=True, with_graph_stats=False,
        use_mesh_feat=False, mesh_tris=None, adj_list=None, g_dir=None, g_filename=None, n_proc=1,
        use_mesh_graph=False, return_graph_nn=False, igraph_nn=None):
    """ This function is developed by Landrieu et al. 
    (see https://github.com/loicland/superpoint_graph). 
    We modified the output of the function to fit our use case. 
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    #---compute 10 nn graph-------
    # target_fea are the indices of the k nearest neighbours of each point (a point itself is not considered as neighbour)
    if igraph_nn is None:
        graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof, verbose=verbose)
        graph_nn = clean_edges_threads(d_mesh=graph_nn)
    #graph_nn = clean_edges(d_mesh=graph_nn)

    #---compute geometric features-------
    if verbose:
        print("Compute geof")
    
    # TODO we could add geodesic features that leverage the mesh structure

    """
    adj_list_ = adj_list.copy()
    xyz = np.array(np.asarray(mesh_vertices_xyz), copy=True)
    rgb = np.array(np.asarray(mesh_vertices_rgb), copy=True)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    tris = np.array(np.asarray(mesh_tris), copy=True)

    mesh_vertices_xyz, mesh_vertices_rgb, mesh_tris, adj_list, 
        lambda_edge_weight=1, reg_strength=0.1, d_se_max=0, k_nn_adj=45, use_cartesian=True, 
        bidirectional=False, respect_direct_neigh=False, n_proc=1, move_vertices=False,
        g_dir=None, g_filename=None, ignore_knn=False, verbose=True, smooth=True, with_graph_stats=False
    """
    if use_mesh_feat or use_mesh_graph:
        d_mesh = get_d_mesh(xyz=xyz, tris=tris, adj_list_=adj_list_, k_nn_adj=k_nn_adj, respect_direct_neigh=False,
            use_cartesian=False, bidirectional=False, n_proc=n_proc, ignore_knn=False, verbose=verbose,
            g_dir=g_dir, g_filename=g_filename)
    if use_mesh_feat
        geof = libply_c.compute_geof(xyz, d_mesh["target"], k_nn_adj, False).astype(np.float32)
    else:
        #geof = libply_c.compute_geof(xyz, graph_nn["c_target"], k_nn_geof, False).astype(np.float32)
        geof = libply_c.compute_geof(xyz, graph_nn["target"], k_nn_adj, False).astype(np.float32)
    if igraph_nn is None:
        del target_fea

    #choose here which features to use for the partition (features vector for each point)
    features = np.hstack((geof, rgb)).astype("float32")#add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
            
    verbosity_level = 0.0
    speed = 2.0
    if use_mesh_graph:
        d_mesh["edge_weight"] = np.array(1. / ( lambda_edge_weight + d_mesh["c_distances"] / np.mean(d_mesh["c_distances"])), dtype = "float32")
        if verbose:
            print("Compute cut pursuit")
        components, in_component, stats = libcp.cutpursuit(features, d_mesh["c_source"], d_mesh["c_target"],
            d_mesh["edge_weight"], reg_strength, 0, 0, 1, verbosity_level, speed)
    else:
        if igraph_nn is None:
            graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["c_distances"] / np.mean(graph_nn["c_distances"])), dtype = "float32")
        #print("        minimal partition...")
        if verbose:
            print("Minimal Partition")
        """
        components: the actual superpoint idxs which is a list of lists (where each list contains point indices)
        in_component: this is one list with the length of number of points - here we got a superpoint idx for each point
            this is just another representation of components.
        """
        components, in_component, stats = libcp.cutpursuit(features, graph_nn["c_source"], graph_nn["c_target"], 
            graph_nn["edge_weight"], reg_strength, 0, 0, 1, verbosity_level, speed)




    stats = stats.tolist()
    stats[0] = int(stats[0]) # n ite main
    stats[1] = int(stats[1]) # iterations
    stats[2] = int(stats[2]) # exit code
    if with_graph_stats:
        geof = features[:, :4] # geometric features
        g_mean = np.mean(geof, axis=0)
        stats.extend(g_mean.tolist())
        g_std = np.std(geof, axis=0)
        stats.extend(g_std.tolist())
        g_med = np.median(geof, axis=0)
        stats.extend(g_med.tolist())
        w_mean = np.mean(graph_nn["edge_weight"])
        stats.append(w_mean)
        w_std = np.std(graph_nn["edge_weight"])
        stats.append(w_std)
        w_median = np.median(graph_nn["edge_weight"])
        stats.append(w_median)
    if verbose:
        print("Done")
        print("Python stats:", stats)
    #print(components)
    #print(in_component)
    # components represent the actual superpoints (partition) - now we need to compute the edges of the superpoints
    components = np.array(components, dtype = "object")
    # the number of components (superpoints) n_com
    n_com = max(in_component)+1

    in_component = np.array(in_component)

    """
    # Uncomment to see how components and in_component are structured
    print("len in_component", in_component.shape)
    #print(np.unique(in_component))
    unis = np.unique(in_component)
    print("in_component")
    for i in range(unis.shape[0]):
        uni = unis[i]
        idxs = np.where(in_component == uni)[0]
        print(uni, len(idxs), idxs)

    #print(unis)
    print("-------------")
    print("components")
    #print(len(components))
    i = 0
    for c in components:
        print(i, len(c), np.sort(c)[:10])
        i += 1
    """
    
    tri = Delaunay(xyz)

    #interface select the edges between different superpoints
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    #edgx and edgxr converts from tetrahedrons to edges (from i to j (edg1) and from j to i (edg1r))
    #done separatly for each edge of the tetrahedrons to limit memory impact
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    # shape: (2, n_tetras)
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    
    del tri, interface
    # edges of points where one end point lies in a superpoint a und another end point in the superpoint b != a
    edges = np.hstack((edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r ,edg5r, edg6r))
    del edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r    
    # Filter edges that occur multiple times in edges, e.g. [[x, x, z], [y, y, z]] -> [[x, z], [y, z]]
    edges = np.unique(edges, axis=1)
    if d_se_max > 0:
        # distance between the superpoints on the edges
        dist = np.sqrt(((xyz[edges[0,:]]-xyz[edges[1,:]])**2).sum(1))
        # filter edges with too large distance
        edges = edges[:,dist<d_se_max]
    
    #---sort edges by alpha numeric order wrt to the components (superpoints) of their source/target---
    # number of edges n_edg
    n_edg = len(edges[0])
    
    # replace point indices in edges with the superpoint idx of each point and save it in edge_comp
    edge_comp = in_component[edges]
    # len(edge_comp_index) == n_edg
    edge_comp_index = n_com * edge_comp[0,:] + edge_comp[1,:]
    order = np.argsort(edge_comp_index)
    
    # reordering of the edges, the edge_comp and the edge_comp_index array itself
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    #print(edge_comp_index)

    """
    np.argwhere(np.diff(edge_comp_index)): where does the sorted edge_comp_index change? 
    This is stored in the array idxs_comp_change. The edge_comp array could have multiple 
    connections between the same superpoints. Therefore, we filter them with the jump_edg 
    in order to take once account of them. Thus, multiple edges from superpoint S_1 to S_2
    will be filtered so that we only have one edge from S_1 to S_2.  
    """
    idxs_comp_change = np.argwhere(np.diff(edge_comp_index)) + 1
    #marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, idxs_comp_change, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1

    ########################################
    # Our modification
    ########################################
    senders = []
    receivers = []
    uni_edges = []
    if verbose:
        print("Total number of edges: {0}".format(n_sedg))
    n_filtered = 0
    if edge_comp.shape[1] == 0:
        raise Exception("Zero edge comp")
    #---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        #ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        #ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        #xyz_source = xyz[ver_source, :]
        #xyz_target = xyz[ver_target, :]
        edge = (com_source, com_target)
        if edge in uni_edges:
            n_filtered += 1
            continue
        senders.append(com_source)
        receivers.append(com_target)
        inv_edge = (com_target, com_source)
        uni_edges.append(inv_edge)
    # bidirectional
    tmp_senders = senders.copy()
    senders.extend(receivers)
    receivers.extend(tmp_senders)
    senders = np.array(senders, dtype=np.uint32)
    receivers = np.array(receivers, dtype=np.uint32)
    if verbose:
        print("{0} edges filtered, {1} unique edges".format(n_filtered, len(uni_edges)))
    if return_graph_nn:
        return n_com, n_sedg, components, senders, receivers, len(uni_edges), stats, graph_nn
    else:
        return n_com, n_sedg, components, senders, receivers, len(uni_edges), stats


def get_characteristic_points(P, center, k, far_points=True):
    """
    Farthest or nearest point sampling. 
    
    Parameters
    ----------
    P : np.ndarray
        The input point cloud where points should be sampled.
    center : np.ndarray
        Center of the point cloud P.
    k : int
        number of points that should be sampled.
    far_points : bool
        If True, farthest point sampling will be applied. Nearest
        point sampling otherwise.

    Returns
    -------
    np.ndarray, np.ndarray
        Indices of the sampled points of P. The distance values
        of the sampled points.
    """
    c_sort_idx = 0
    if far_points:
        c_sort_idx = -1
    d_sort, distances = distance_sort(P=P, p_query=center)
    center_idx = int(d_sort[0])

    c_idx = center_idx
    exclude = [c_idx]
    idxs = []
    dists = []
    n_P = P.shape[0]
    if k > n_P:
        k = n_P - 1
    for i in range(k):
        d_sort, distances = distance_sort(P=P, p_query=P[c_idx])
        
        # apply mask
        mask = np.ones((d_sort.shape[0], ), dtype=np.bool)
        for j in range(len(exclude)):
            tmp_idx = np.where(d_sort == exclude[j])[0]
            mask[tmp_idx] = False
        d_sort_ = d_sort[mask]
        try:
            c_idx = int(d_sort_[c_sort_idx])
        except:
            break
        c_dist = distances[c_idx]
        idxs.append(c_idx)
        dists.append(c_dist)
        if c_idx in exclude:
            raise Exception("Sampling error")
        exclude.append(c_idx)
    return idxs, np.array(dists, dtype=np.float32)


def zero_padding(x, target_size):
    """Append zeros to a vector or an nd array (in the last dimension).

    Parameters
    ----------
    x : np.ndarray
        Input points that should be padded with zeros.
    target_size : int
        Target length of the last dimension.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    if x.shape[0] < target_size:
        diff_target = target_size - x.shape[0]
        if len(x.shape) <= 1:
            return np.vstack( (x[:, None], np.zeros((diff_target, 1)) ) ).reshape(target_size, )
        else:
            target_shape = (diff_target, ) + x.shape[1:]
            #print(target_shape, target_size)
            return np.vstack( (x, np.zeros(target_shape) ) )
    else:
        return x


def get_volume(bb, off=0):
    """Calculates the volume of a bounding box in 3 dimensions.

    Parameters
    ----------
    bb : np.ndarray
        Bounding box.
    off : int
        Skip dimensions with off > 0.

    Returns
    -------
    float
        Volume of the bounding box.
    """
    volume = 1
    for i in [1,3,5]:
        volume *= (bb[i+off] - bb[i-1+off])
    return volume


def get_min_max(feature, target, pad=True):
    """Get the minimum and maximum values of a feature
    vector. 

    Parameters
    ----------
    feature : np.ndarray
        Feature vector where the minimum and maximum should be calculated. 
    target : int
        Target length of the resulting vector. One half will consist of the min
        values and the other half of the max values. 
    pad : bool
        Should zero padding be applied?

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        An array of the minimum and the maximum features with the
        corresponding indices. 
    """
    sortation = np.argsort(feature)
    t_half = int(target / 2)
    half = t_half
    if sortation.shape[0] < target:
        if sortation.shape[0] % 2 == 0:
            half = sortation.shape[0]
        else:
            half = sortation.shape[0] - 1
        half = int(half / 2)
    
    min_idxs = sortation[:half]
    max_idxs = sortation[-half:]

    min_f = feature[min_idxs]
    max_f = feature[max_idxs]
    if pad:
        min_f = zero_padding(x=min_f, target_size=t_half)
        max_f = zero_padding(x=max_f, target_size=t_half)
    return min_f, max_f, min_idxs, max_idxs


def extract_min_max_points(P, min_idxs, max_idxs, target, center=None, center_max=None):
    """Extract the points according to indices where a
    features has minimum and maximum values.

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    min_idxs : np.ndarray
        Indices of the minimum of a feature.
    max_idxs : np.ndarray
        Indices of the maximum of a feature.
    target : int
        Target number of points.
    center : np.ndarray
        Center of the point cloud P.
    center_max : int
        Maximum index to determine the center point. It can be used to 
        ignore translate points into the origin with center_max=3.

    Returns
    -------
    np.ndarray, np.ndarray
        Minimum and the maximum points.
    """
    min_P = P[min_idxs]
    max_P = P[max_idxs]
    if center is not None and center_max is not None:
        min_P[:, :center_max] -= center[:center_max]
        max_P[:, :center_max] -= center[:center_max]
    #print(min_P.shape, max_P.shape)
    min_P = zero_padding(x=min_P, target_size=target)
    max_P = zero_padding(x=max_P, target_size=target)
    min_P = min_P.flatten()
    max_P = max_P.flatten()
    return min_P, max_P


def hists_feature(P, center, bins=10, min_r=-0.5, max_r=0.5):
    """Create a histogram of the feature values of a point cloud.

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    center : np.ndarray
        Center of the point cloud P.
    bins : int
        Number of bins to create the histogram.
    min_r : float
        Minimum value of the range to compute the histogram.
    max_r : float
        Maximum value of the range to compute the histogram.

    Returns
    -------
    np.ndarray
        Histogram values for every dimension. Output shape is the
        number of dimensions times bins. 
    """
    hists = np.zeros((P.shape[1] * bins))
    for i in range(P.shape[1]):
        hist, bin_edges = np.histogram(a=P[:, i]-center[i], bins=bins, range=(min_r, max_r))
        start = i * bins
        stop = start + bins
        hists[start:stop] = hist / P.shape[0]
    return hists


def compute_features(cloud, n_curv=30, k_curv=14, k_far=30, n_normal=30, bins=10, min_r=-0.5, max_r=0.5):
    """Compute features from a point cloud P. The features of a point cloud are:
    - Mean color 
    - Median color
    - 25% Quantil of the color
    - 75% Quantil of the color
    - Standard deviation of all features
    - Average normal
    - Standard deviation of the normals
    - Median normal
    - 25% Quantil of the normals
    - 75% Quantil of the normals
    - n_normal/2 maximum angles of the normals with the x-axis
    - n_normal/2 minimum angles of the normals with the x-axis
    - n_normal/2 points that correspond to the max normal angle values
    - n_normal/2 points that correspond to the min normal angle values
    - Average curvature value
    - Standard deviation of the curvature values
    - Median curvature value
    - 25% Quantil of the curvature values
    - 75% Quantil of the curvature values
    - n_curv/2 maximum curvature values
    - n_curv/2 minimum curvature values
    - n_curv/2 points that correspond to the max curvature values
    - n_curv/2 points that correspond to the min curvature values
    - k_far farthest points
    - k_far distances of the farthest points
    - volume of the spatial bounding box
    - volume of the color bounding box
    - histograms of every dimension

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    n_curv : int
        Number of points for the curvature feature.
    k_curv : int
        Number of neighbours in order to calculate the  curvature features.
    k_far : int
        Number of points to sample the farthest points.  
    n_normal : int
        Number of normal features.
    bins : int
        Number of bins for a histogram in every dimension of the point cloud.
    min_r : float
        Minimum value to consider to calculate the histogram.
    max_r : float
        Maximum value to consider to calculate the histogram.


    Returns
    -------
    np.ndarray
        The calculated features.
    """
    P = cloud
    center = np.mean(P, axis=0)
    std = np.std(P, axis=0)
    median_color = np.median(P[:, 3:], axis=0)
    q_25_color = np.quantile(P[:, 3:], 0.25, axis=0)
    q_75_color = np.quantile(P[:, 3:], 0.75, axis=0)
    
    normals, curv, ok = estimate_normals_curvature(P=P[:, :3], k_neighbours=k_curv)
    
    ########normals########
    # mean, std
    if ok:
        mean_normal = np.mean(normals, axis=0)
        std_normal = np.std(normals, axis=0)
        median_normal = np.median(normals, axis=0)
        q_25_normal = np.quantile(normals, 0.25, axis=0)
        q_75_normal = np.quantile(normals, 0.75, axis=0)
        #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_normal.shape, std_normal.shape, median_normal.shape, q_25_normal.shape, q_75_normal.shape))

        # min, max
        ref_normal = np.zeros((normals.shape[0], 3))
        ref_normal[:, 0] = 1
        normal_dot, _, _ = np_fast_dot(a=ref_normal, b=normals)
        min_normal, max_normal, min_n_idxs, max_n_idxs = get_min_max(feature=normal_dot, target=n_normal)
        min_P_n, max_P_n = extract_min_max_points(P=P, min_idxs=min_n_idxs, max_idxs=max_n_idxs, target=int(n_normal/2), center=center, center_max=3)
        #print("min f: {0}\nmax f: {1}".format(min_normal.shape, max_normal.shape))
        #print("min P: {0}\nmax P: {1}".format(min_P_n.shape, max_P_n.shape))

        ########curvature########
        # mean, std
        curv = curv.reshape(curv.shape[0], )
        mean_curv = 2 * (np.mean(curv) - 0.5)
        mean_curv = np.array([mean_curv], dtype=np.float32)
        std_curv = np.std(curv)
        std_curv = np.array([std_curv], dtype=np.float32)
        median_curv = np.median(curv)
        median_curv = 2 * (median_curv - 0.5)
        median_curv = np.array([median_curv], dtype=np.float32)
        q_25_curv = np.quantile(curv, 0.25)
        q_25_curv = 2 * (q_25_curv - 0.5)
        q_25_curv = np.array([q_25_curv], dtype=np.float32)
        q_75_curv = np.quantile(curv, 0.75)
        q_75_curv = 2 * (q_75_curv - 0.5)
        q_75_curv = np.array([q_75_curv], dtype=np.float32)
        #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_curv.shape, std_curv.shape, median_curv.shape, q_25_curv.shape, q_75_curv.shape))
        
        # min, max
        min_curv, max_curv, min_c_idxs, max_c_idxs = get_min_max(feature=curv, target=n_curv, pad=False)
        min_curv = 2 * (min_curv - 0.5)
        max_curv = 2 * (max_curv - 0.5) 
        target_c = int(n_curv / 2)
        min_curv = zero_padding(x=min_curv, target_size=target_c)
        max_curv = zero_padding(x=max_curv, target_size=target_c)
        min_P_c, max_P_c = extract_min_max_points(P=P, min_idxs=min_c_idxs, max_idxs=max_c_idxs, target=target_c, center=center, center_max=3)
        #print("min f: {0}\nmax f: {1}".format(min_curv.shape, max_curv.shape))
        #print("min P: {0}\nmax P: {1}".format(min_P_c.shape, max_P_c.shape))
    else:
        mean_normal = np.zeros((3, ))
        std_normal = np.zeros((3, ))
        median_normal = np.zeros((3, ))
        q_25_normal = np.zeros((3, ))
        q_75_normal = np.zeros((3, ))
        #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_normal.shape, std_normal.shape, median_normal.shape, q_25_normal.shape, q_75_normal.shape))
        n_normal_ = int(n_normal/2)

        min_normal = np.zeros((n_normal_, ))
        max_normal = np.zeros((n_normal_, ))
        
        min_P_n = np.zeros((6*n_normal_, ))
        max_P_n = np.zeros((6*n_normal_, ))
        #print("min f: {0}\nmax f: {1}".format(min_normal.shape, max_normal.shape))
        #print("min P: {0}\nmax P: {1}".format(min_P_n.shape, max_P_n.shape))

        mean_curv = np.zeros((1, ))
        std_curv = np.zeros((1, ))
        median_curv = np.zeros((1, ))
        q_25_curv = np.zeros((1, ))
        q_75_curv = np.zeros((1, ))
        #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_curv.shape, std_curv.shape, median_curv.shape, q_25_curv.shape, q_75_curv.shape))
        n_curv_ = int(n_curv/2)
        min_curv = np.zeros((n_curv_, ))
        max_curv = np.zeros((n_curv_, ))
        min_P_c = np.zeros((6*n_curv_, ))
        max_P_c = np.zeros((6*n_curv_, ))
        #print("min f: {0}\nmax f: {1}".format(min_curv.shape, max_curv.shape))
        #print("min P: {0}\nmax P: {1}".format(min_P_c.shape, max_P_c.shape))

    ########farthest points########
    idxs, dists = get_characteristic_points(P=P, center=center, k=k_far, far_points=True)
    far_points = P[idxs]
    far_points[:, :3] -= center[:3]
    far_points = zero_padding(x=far_points, target_size=k_far)
    far_points = far_points.flatten()
    dists = zero_padding(x=dists, target_size=k_far)

    ########volumes########
    bb = get_general_bb(P=P)
    spatial_volume = get_volume(bb=bb, off=0)
    color_volume = get_volume(bb=bb, off=6)
    volumes = np.array([spatial_volume, color_volume])

    ########hists########
    hists = hists_feature(P=P, center=center, bins=bins, min_r=min_r, max_r=max_r)
    hists -= 0.5
    hists *= 2

    ########concatenation########
    features = np.vstack(
        (
        center[3:, None],#3,abs
        median_color[:, None],#3,abs
        q_25_color[:, None],#3,abs
        q_75_color[:, None],#3,abs
        std[:, None],#6
        
        mean_normal[:, None],#3
        std_normal[:, None],#3
        median_normal[:, None],#3,abs
        q_25_normal[:, None],
        q_75_normal[:, None],
        
        min_normal[:, None],#n_normal*3
        max_normal[:, None],#n_normal*3
        
        min_P_n[:, None],#n_normal*6,local
        max_P_n[:, None],#n_normal*6,local
        
        mean_curv[:, None],#1
        std_curv[:, None],#1
        median_curv[:, None],#3,abs
        q_25_curv[:, None],
        q_75_curv[:, None],
        min_curv[:, None],#n_curv/2
        max_curv[:, None],#n_curv/2
        min_P_c[:, None],#6*n_curv/2,local
        max_P_c[:,None],#6*n_curv/2,local
        
        far_points[:, None],#k_far*6,local
        dists[:, None],#k_far
        volumes[:, None],#2
        hists[:, None]#6*bins
        ))
    features = features.astype(dtype=np.float32)
    features = features.reshape(features.shape[0], )
    return features


def np_fast_dot(a, b):
    dot = a*b
    dot = np.sum(dot, axis=-1)
    a_n = np.linalg.norm(a, axis=-1)
    b_n = np.linalg.norm(b, axis=-1)
    dot /= ((a_n * b_n) + 1e-6)
    return dot, a_n, b_n


def distance_sort(P, p_query):
    d_points = P[:, :3] - p_query[:3]
    d_square = np.square(d_points)
    d = np.sum(d_square, axis=1)
    d_sort = np.argsort(d)
    return d_sort, d


def estimate_normals_curvature(P, k_neighbours=5):
    n_P = P.shape[0]
    if n_P <= k_neighbours:
        k_neighbours = 5
    n = P.shape[1]

    try:
        nbrs = NearestNeighbors(n_neighbors=k_neighbours, algorithm="brute", metric="euclidean").fit(P[:, :3])
        _, nns = nbrs.kneighbors(P[:, :3])
    except Exception as e:
        normals = np.zeros((n_P, 3), dtype=np.float32)
        curvature = np.zeros((n_P, ), dtype=np.float32)
        return normals, curvature, False


    k_nns = nns[:, 1:k_neighbours]
    p_nns = P[k_nns[:]]
    
    p = np.matlib.repmat(P, k_neighbours-1, 1)
    p = np.reshape(p, (n_P, k_neighbours-1, n))
    p = p - p_nns
    
    C = np.zeros((n_P,6))
    C[:,0] = np.sum(np.multiply(p[:,:,0], p[:,:,0]), axis=1)
    C[:,1] = np.sum(np.multiply(p[:,:,0], p[:,:,1]), axis=1)
    C[:,2] = np.sum(np.multiply(p[:,:,0], p[:,:,2]), axis=1)
    C[:,3] = np.sum(np.multiply(p[:,:,1], p[:,:,1]), axis=1)
    C[:,4] = np.sum(np.multiply(p[:,:,1], p[:,:,2]), axis=1)
    C[:,5] = np.sum(np.multiply(p[:,:,2], p[:,:,2]), axis=1)
    C /= k_neighbours
    
    normals = np.zeros((n_P,n))
    curvature = np.zeros((n_P,1))
    
    for i in range(n_P):
        C_mat = np.array([[C[i,0], C[i,1], C[i,2]],
            [C[i,1], C[i,3], C[i,4]],
            [C[i,2], C[i,4], C[i,5]]])
        values, vectors = np.linalg.eig(C_mat)
        lamda = np.min(values)
        k = np.argmin(values)
        norm = np.linalg.norm(vectors[:,k])
        if norm == 0:
            norm = 1e-12
        normals[i,:] = vectors[:,k] / np.linalg.norm(vectors[:,k])
        sum_v = np.sum(values)
        if sum_v == 0:
            sum_v = 1e-12
        curvature[i] = lamda / sum_v

    return normals, curvature, True


def get_general_bb(P):
    bb = np.zeros((2*P.shape[1], ))
    j = 0
    for i in range(P.shape[1]):
        bb[j] = np.min(P[:, i])
        j += 1
        bb[j] = np.max(P[:, i])
        j += 1
    return bb


def feature_point_cloud(P):
    center = np.mean(P[:, :3], axis=0)
    max_P = np.max(np.abs(P[:, :3]))
    P[:, :3] -= center
    P[:, :3] /= max_P
    P[:, 3:] /= 255
    P[:, 3:] -= 0.5
    P[:, 3:] *= 2
    return P, center


def graph(cloud, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.1, d_se_max=0, max_sp_size=-1):
    """ This function creates a superpoint graph from a point cloud. 

    Parameters
    ----------
    cloud : np.ndarray
        A point cloud with the columns xyzrgb.
    k_nn_adj : int
        TODO.
    k_nn_adj : int
        TODO.
    lambda_edge_weight : float
        TODO
    reg_strength : flaot
        TODO
    d_se_max : float
        TODO
    max_sp_size : int
        Maximum size of a superpoint. 
    
    Returns
    -------
    dict, list[np.ndarray]
        The graph dictionary which is used by the neural network.
        A list of point indices for each superpoint.  
    """
    # make a copy of the original cloud to prevent that points do not change any properties
    P = np.array(cloud, copy=True)
    print("Compute superpoint graph")
    t1 = time.time()
    n_sps, n_edges, sp_idxs, senders, receivers, _ = superpoint_graph(
        xyz=P[:, :3],
        rgb=P[:, 3:],
        reg_strength=reg_strength,
        k_nn_geof=k_nn_geof,
        k_nn_adj=k_nn_adj,
        lambda_edge_weight=lambda_edge_weight,
        d_se_max=d_se_max)

    """
    n_sps: Number of superpoints (vertices) in the graph
    n_edges: Number of edges in the graph
    sp_idxs: A list of point indices for each superpoint
    senders: List of start vertices for each edge in a directed graph
    receivers: List of end vertices for each edge in a directed graph
    """
    #############################################
    # create a feature vector for every superpoint
    t2 = time.time()
    duration = t2 - t1
    print("Superpoint graph has {0} nodes and {1} edges (duration: {2:.3f} seconds)".format(n_sps, n_edges, duration))

    print("Compute features for every superpoint")

    t1 = time.time()
    P, center = feature_point_cloud(P=P)

    print("Check superpoint sizes")
    # so that every superpoint is smaller than the max_sp_size
    sps_sizes = []
    for i in range(n_sps):
        idxs = np.unique(sp_idxs[i])
        sp_idxs[i] = idxs
        sp_size = idxs.shape[0]
        if max_sp_size > 0 and sp_size > max_sp_size:
            raise Exception("Superpoint {0} too large with {1} points (max: {2}). Try to lower the reg_strength.".format(i, sp_size, max_sp_size))
        sps_sizes.append(sp_size)
    print("Average superpoint size: {0:.2f} ({1:.2f})".format(np.mean(sps_sizes), np.std(sps_sizes)))

    idxs = sp_idxs[0]
    sp = P[idxs]
    features = compute_features(cloud=sp)
    n_ft = features.shape[0]
    print("Use {0} features".format(n_ft))

    node_features = np.zeros((n_sps, n_ft), dtype=np.float32)
    node_features[0] = features
    for k in tqdm(range(1, n_sps), desc="Node features"):
        idxs = sp_idxs[k]
        sp = P[idxs]
        features = compute_features(cloud=sp)
        node_features[k] = features

    t2 = time.time()
    duration = t2 - t1
    print("Computed features in {0:.3f} seconds".format(duration))

    graph_dict = {
        "nodes": node_features,
        "senders": senders,
        "receivers": receivers,
        "edges": None,
        "globals": None
        }
    return graph_dict, sp_idxs


def init_model(n_ft, is_mesh=False):
    """Initialize the neural network.

    Parameters
    ----------
    n_ft : int
        Number of features n_ft that are used by the neural network

    Returns
    -------
    model : network.FFGraphNet
        The model that predicts the correlations
    """
    model = FFGraphNet(
        name="target_policy",
        n_ft_outpt=8,
        n_actions=2,
        seed=42,
        trainable=True,
        check_numerics=False,
        initializer="glorot_uniform",
        mode="full",
        stateful=False,
        discrete=True,
        head_only=True,
        observation_size=(910, ))
    n_nodes = 6
    features_points = 6
    feature_func = compute_features
    if is_mesh:
        features_points = 9
        feature_func = compute_mesh_features
    node_features = np.zeros((n_nodes, n_ft), dtype=np.float32)
    for i in range(n_nodes):
        sp = np.random.randn(100, features_points)
        features = feature_func(cloud=sp)
        features = features.reshape(features.shape[0], )        
        node_features[i] = features
    senders = np.array(list(range(0, n_nodes-1)), dtype=np.uint32)
    receivers = np.array(list(range(1, n_nodes)), dtype=np.uint32)
    graph_dict = {
        "nodes": node_features,
        "senders": senders,
        "receivers": receivers,
        "edges": None,
        "globals": None
    }
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    model.action(obs=input_graphs, training=False)
    model.load(directory="./", filename="model")
    return model


def predict(graph_dict, dec_b=0.5, is_mesh=False):
    """Predict correlations between the nodes in the graph

    Parameters
    ----------
    graph_dict : dict()
        Superpoint graph.
    dec_b : float
        Decision boundary in the range between [0,1].
    is_mesh : bool
        Is a mesh subject of the prediction?

    Returns
    -------
    action : np.ndarray
        Which nodes in the graph should be unified.
    probs : np.ndarray
        Correlations between the nodes

    """
    t1 = time.time()
    n_ft = graph_dict["nodes"].shape[1]
    model = init_model(n_ft=n_ft, is_mesh=is_mesh)
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    
    a_dict = model.action(obs=input_graphs, training=False, decision_boundary=dec_b)
    
    action = a_dict["action"]
    action = action.numpy()
    action = action.astype(np.bool)
    
    probs = a_dict["probs"]
    probs = probs.numpy()
    t2 = time.time()
    duration = t2 - t1
    print("Model prediction takes {0:.3f} seconds".format(duration))
    return action, probs


def c(k, sp_idxs, P):
    # TODO can be deleted
    idxs = sp_idxs[k]
    sp = P[idxs]
    features = compute_mesh_features(cloud=sp)
    return features


def compute_mesh_features(cloud, k_far=30, n_normal=30, bins=10, min_r=-0.5, max_r=0.5):
    """Compute features from mesh cloud. The features of a point cloud are:
    - Mean color 
    - Median color
    - 25% Quantil of the color
    - 75% Quantil of the color
    - Standard deviation of all features
    - Average normal
    - Standard deviation of the normals
    - Median normal
    - 25% Quantil of the normals
    - 75% Quantil of the normals
    - n_normal/2 maximum angles of the normals with the x-axis
    - n_normal/2 minimum angles of the normals with the x-axis
    - n_normal/2 points that correspond to the max normal angle values
    - n_normal/2 points that correspond to the min normal angle values
    - k_far farthest points
    - k_far distances of the farthest points
    - volume of the spatial bounding box
    - volume of the color bounding box
    - histograms of every dimension

    Parameters
    ----------
    P : np.ndarray
        The input point cloud.
    k_far : int
        Number of points to sample the farthest points.  
    n_normal : int
        Number of normal features.
    bins : int
        Number of bins for a histogram in every dimension of the point cloud.
    min_r : float
        Minimum value to consider to calculate the histogram.
    max_r : float
        Maximum value to consider to calculate the histogram.


    Returns
    -------
    np.ndarray
        The calculated features.
    """
    P = cloud[:, :6]
    normals = cloud[:, 6:9]
    center = np.mean(P, axis=0)
    std = np.std(P, axis=0)
    median_color = np.median(P[:, 3:], axis=0)
    q_25_color = np.quantile(P[:, 3:], 0.25, axis=0)
    q_75_color = np.quantile(P[:, 3:], 0.75, axis=0)
    
    ########normals########
    # mean, std
    mean_normal = np.mean(normals, axis=0)
    std_normal = np.std(normals, axis=0)
    median_normal = np.median(normals, axis=0)
    q_25_normal = np.quantile(normals, 0.25, axis=0)
    q_75_normal = np.quantile(normals, 0.75, axis=0)
    #print("mean: {0}\nstd: {1}\nmedian: {2}\nq25: {3}\nq75: {4}".format(mean_normal.shape, std_normal.shape, median_normal.shape, q_25_normal.shape, q_75_normal.shape))

    # min, max
    ref_normal = np.zeros((normals.shape[0], 3))
    ref_normal[:, 0] = 1
    normal_dot, _, _ = np_fast_dot(a=ref_normal, b=normals)
    min_normal, max_normal, min_n_idxs, max_n_idxs = get_min_max(feature=normal_dot, target=n_normal)
    min_P_n, max_P_n = extract_min_max_points(P=P, min_idxs=min_n_idxs, max_idxs=max_n_idxs, target=int(n_normal/2), center=center, center_max=3)
    #print("min f: {0}\nmax f: {1}".format(min_normal.shape, max_normal.shape))
    #print("min P: {0}\nmax P: {1}".format(min_P_n.shape, max_P_n.shape))

    ########farthest points########
    idxs, dists = get_characteristic_points(P=P, center=center, k=k_far, far_points=True)
    far_points = P[idxs]
    far_points[:, :3] -= center[:3]
    far_points = zero_padding(x=far_points, target_size=k_far)
    far_points = far_points.flatten()
    dists = zero_padding(x=dists, target_size=k_far)

    ########volumes########
    bb = get_general_bb(P=P)
    spatial_volume = get_volume(bb=bb, off=0)
    color_volume = get_volume(bb=bb, off=6)
    volumes = np.array([spatial_volume, color_volume])

    ########hists########
    hists = hists_feature(P=P, center=center, bins=bins, min_r=min_r, max_r=max_r)
    hists -= 0.5
    hists *= 2

    ########concatenation########
    features = np.vstack(
        (
        center[3:, None],#3,abs
        median_color[:, None],#3,abs
        q_25_color[:, None],#3,abs
        q_75_color[:, None],#3,abs
        std[:, None],#6
        
        mean_normal[:, None],#3
        std_normal[:, None],#3
        median_normal[:, None],#3,abs
        q_25_normal[:, None],
        q_75_normal[:, None],
        min_normal[:, None],#n_normal*3
        max_normal[:, None],#n_normal*3
        
        min_P_n[:, None],#n_normal*6,local
        max_P_n[:, None],#n_normal*6,local
        
        far_points[:, None],#k_far*6,local
        dists[:, None],#k_far
        volumes[:, None],#2
        hists[:, None]#6*bins
        ))
    features = features.astype(dtype=np.float32)
    #features = features.reshape(features.shape[0], )
    return features


def get_neighbourhood(c, direct_neigh_idxs, n_edges, edges, distances, visited):
    """
    Parameters
    ----------
    c : int
        Vertex index (in reference to the graph) of the currently
        closest vertex to the start vertex 's'.
    edges : np.array(int)
        2X|E| array that stores the edges. The first row represents the
        start vertex and the second row the target vertices of an edge. 
        The array should be sorted according to the start vertices.  
    direct_neigh_idxs : np.array(int)
        Indices to the first target vertex in the 'edges' array. 
    n_edges : np.array(int)
        Number of edges that origin from a start vertex. 
    distances : np.array(float)
        The weight of each edge (e.g. the spatial distance) 
    visited : list(int)
        Vertices that are already considered as closest vertices to 's'
    """
    # get the neighbour vertices of 'c'
    start_i = direct_neigh_idxs[c]
    stop_i = start_i + n_edges[c]
    N_c = edges[1, start_i:stop_i]
    # and the corresponding distances from 's'
    d_N = distances[start_i:stop_i]
    # filter vertices which we already considered
    N_c_ = []
    d_N_ = []
    for j in range(len(N_c)):
        i = N_c[j]
        if i in visited:
            continue
        # only append vertices that we have not visited yet
        N_c_.append(i)
        d_N_.append(d_N[j])
    return N_c_, d_N_


def get_next(R, d_R):
    """
    Parameters
    ----------
    R : list(int)
        List of vertex indices (in reference to the graph) of vertices
        which can be considered as next closest vertices to 's'.
    d_R : list(float)
        The distances of the vertices in 'R'.
    """
    R = np.array(R)
    d_R = np.array(d_R)
    # sort 'R' and 'd_R' according to the distances from 's'
    sortation = np.argsort(d_R)
    R = R[sortation].tolist()
    d_R = d_R[sortation].tolist()
    
    # take the closest vertex to 's' and the distance 'd_sc'
    c = R[0]
    d_sc = d_R[0]
    # delete them as points that can be considered second closest vertices 
    del R[0]
    del d_R[0]

    # assign the second closest vertex 'n' and its distance 'd_sn'
    n = R[0]
    d_sn = d_R[0]
    return c, d_sc, n, d_sn, R, d_R


def search(c, d_sc, n, d_sn, R, d_R, t, N_f, d_f, 
        direct_neigh_idxs, n_edges, edges, distances):  
    """
    Parameters
    ----------
    c : int
        Vertex index (in reference to the graph) of the currently
        closest vertex to the start vertex 's'.
    d_sc : float
        The geodesic distance between 's' and 'c'.
    n : int
        Vertex index (in reference to the graph) of the currently
        second closest vertex to 's'.
    d_sn : float
        The geodesic distance between 's' and 'n'.
    R : list(int)
        List of vertex indices (in reference to the graph) of vertices
        which can be considered as next closest vertices to 's'.
    d_R : list(float)
        The distances of the vertices in 'R'.
    t : int
        The number of neighbours that should be found.
    N_f : list(int)
        The final vertex indices (in reference to the graph) of the closest
        vertices to 's'.
    d_f : list(float)
        The distances of the vertices in 'N_f'.
    edges : np.array(int)
        2X|E| array that stores the edges. The first row represents the
        start vertex and the second row the target vertices of an edge. 
        The array should be sorted according to the start vertices.  
    direct_neigh_idxs : np.array(int)
        Indices to the first target vertex in the 'edges' array. 
    n_edges : np.array(int)
        Number of edges that origin from a start vertex. 
    distances : np.array(float)
        The weight of each edge (e.g. the spatial distance) 
    """
    len_R = len(R)
    # we do not want to visit a vertex 'c' twice
    if c in N_f:
        # if we can determine a best and a second best vertex
        if len_R > 1:
            c, d_sc, n, d_sn, R, d_R = get_next(R=R, d_R=d_R)
            search(
                c=c, d_sc=d_sc, n=n, d_sn=d_sn, R=R, d_R=d_R, t=t, 
                N_f=N_f, d_f=d_f, direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges, edges=edges, distances=distances)
        elif len_R == 1:
            # add the last remaining vertex
            N_f.append(R[0])
            d_f.append(d_R[0])
        return
    # append the current vertex 'c' to our final list as it is the clostest vertex
    N_f.append(c)
    d_f.append(d_sc)
    # if we have enough neighbours or no second best vertex can be determined
    if len(N_f) >= t or len_R <= 1:
        ###########WE DO NOT NEED THIS?########################
        #if len_R == 1:
            # add the last remaining vertex
            #N_f.append(R[0])
            #d_f.append(d_R[0])
        ###########WE DO NOT NEED THIS?########################
        return
    # get the neighbour vertices of 'c' and their distances to c ('d_N')
    N_c, d_N = get_neighbourhood(
                c=c,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                edges=edges,
                distances=distances,
                visited=N_f)
    len_N_c = len(N_c)
    # in case of a empty neighbourhood
    if len_N_c == 0:
        if len_R > 1:
            c, d_sc, n, d_sn, R, d_R = get_next(R=R, d_R=d_R)
            search(
                c=c, d_sc=d_sc, n=n, d_sn=d_sn, R=R, d_R=d_R, t=t, 
                N_f=N_f, d_f=d_f, direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges, edges=edges, distances=distances)
        elif len_R == 1:
            # add the last remaining vertex
            N_f.append(R[0])
            d_f.append(d_R[0])
        return
    """
    determine if a neighbour 'i' is closer to the start vertex than the
    second closest vertex 'n'
    """

    """
    flag to indicate wether the second best vertex should be updated
    this is the case if every neighbour vertex of 'c' is more far away
    from the start vertex 's' than the second best neigbour 'n'
    """
    u = True
    # vertex index (in the graph) of the closest
    next_ = None
    # container to determine the closest neighbour 'i' to the start vertex
    # distance of the closest 
    d_next = np.inf
    # vertex index (in the neighbourhood of 'c') of the closest
    i_next = None
    for j in range(len_N_c):
        i = N_c[j]
        d_ci = d_N[j]
        """
        the distance from the start vertex 's' to a vertex 'i' is the distance
        from the start vertex 's' to the current vertex 'c' ('d_sc') plus the distance
        from the current vertex 'c' to a neighbour vertex 'i' ('d_ci') 
        """ 
        d_si = d_sc + d_ci
        # update the distance of that neighbour
        d_N[j] = d_si
        # if the neighbour is closer to the start vertex than the second closest vertex 'n'
        if d_si <= d_sn:
            # at least one neighbour 'i' is closer to the start vertex 's' than 'n'
            u = False
            # determine which neighbour is closest to 's'
            if d_si < d_next:
                d_next = d_si
                next_ = i
                i_next = j
    # if the 'n' is closer to 's' than all neighbours of 'c'
    if u:
        # set 'n' as 'c' and determine the next best vertex
        R.extend(N_c)
        d_R.extend(d_N)
        c, d_sc, n, d_sn, R, d_R = get_next(R=R, d_R=d_R)
    else:
        # we do not need to consider the closest 'i'-th neighbour anymore
        del N_c[i_next]
        del d_N[i_next]
        # the remaining neighbourhood can be considered as future closest vertices
        R.extend(N_c)
        d_R.extend(d_N)
        # assign the closest 'i'-th neighbour as 'c'
        c = next_
        d_sc = d_next
    search(
        c=c, d_sc=d_sc, n=n, d_sn=d_sn, R=R, d_R=d_R, t=t,
        N_f=N_f, d_f=d_f, direct_neigh_idxs=direct_neigh_idxs,
        n_edges=n_edges, edges=edges, distances=distances)


def f(v_idx, knn, edges, direct_neigh_idxs, n_edges, distances, v, tree):
    N_f = [v_idx]
    d_f = [0]
    N_c, d_N = get_neighbourhood(
        c=v_idx,
        direct_neigh_idxs=direct_neigh_idxs,
        n_edges=n_edges,
        edges=edges,
        distances=distances,
        visited=N_f)
    R = N_c.copy()
    d_R = d_N.copy()
    c, d_sc, n, d_sn, R, d_R = get_next(R=R, d_R=d_R)
    search(
        c=c, d_sc=d_sc, n=n, d_sn=d_sn, R=R, d_R=d_R, t=knn+1, N_f=N_f,
        d_f=d_f, direct_neigh_idxs=direct_neigh_idxs,
        n_edges=n_edges, edges=edges, distances=distances)
    d_f = np.array(d_f[1:])
    N_f = np.array(N_f[1:])
    sortation = np.argsort(d_f)
    N_f = N_f[sortation]
    d_f = d_f[sortation]

    f_edges = np.zeros((3, knn), dtype=np.float32)
    f_edges[0, :] = v_idx
    vi = 0
    stop_i = knn

    if len(N_f) < knn:
        #print("test", len(N_f))
        f_edges[1, vi:vi+len(N_f)] = N_f
        # we have found less neighbours than knn
        missing = knn - len(N_f)
        # search the euclidean nearest neighbours which we haven't considered as neighbours yet
        nn_distances, nn_indices = tree.query(x=v, k=knn+1)
        nn_distances = nn_distances[1:]
        nn_indices = nn_indices[1:]
        # filter vertices that we already consider as neighbours
        add_n = []
        add_d = []
        for j in range(len(nn_indices)):
            idx = nn_indices[j]
            if idx in N_f:
                # we already consider this vertex as neighbour
                continue
            add_n.append(idx)
            add_d.append(nn_distances[j])
            # check if we have enough neighbours
            if len(add_n) == missing:
                break
        if len(add_n) < missing:
            raise Exception("Choose another k: {0}".format(knn))
        if missing == 1:
            # we only need to set one neighbour
            # determine the index of that neighbour
            vix = stop_i - 1
            # save the edge and the distance
            f_edges[1, vix] = add_n[0]
            f_edges[2, vix] = add_d[0]
        else:
            # we need to set the 'missing neighbours'
            # determine the interval where have to set the neighbours
            vi_start = stop_i - missing
            # save the edges and the distances
            f_edges[1, vi_start:] = add_n
            f_edges[2, vi_start:] = add_d
    else:
        f_edges[1, :] = N_f[:knn]
        f_edges[2, :] = d_f[:knn]
    return f_edges


def get_neigh(v, dv, edges, direct_neigh_idxs, n_edges, distances):
    """Query the direct neighbours of the vertex v. The distance dv to the vertex v
    will be added to the distances of the direct neigbours. 

    Parameters
    ----------
    v : int
        A vertex index
    dv : float
        Distances to the vertex v
    edges : np.ndarray
        Edges in the graph in a source target format
    direct_neigh_idxs : np.ndarray
        Array with the direct neighbours of the vertices in the graph.
    n_edges : np.ndarray
        Number of adjacent vertices per vertex
    distances : np.ndarray
        Array that containes the direct neigbours of a vertex

    Returns
    -------
    neighs : np.ndarray
        Direct neighbours (adjacent vertices) of the vertex v
    dists : np.ndarray
        The distances to the direct neighbourhood.
    """
    start = direct_neigh_idxs[v]
    stop = start + n_edges[v]
    neighs = edges[1, start:stop]
    dists = dv + distances[start:stop]
    return neighs, dists


def search_bfs(vi, edges, distances, direct_neigh_idxs, n_edges, k):
    """Search k nearest neigbours of a vertex with index vi with a BFS.

    Parameters
    ----------
    vi : int
        A vertex index
    edges : np.ndarray
        The edges of the mesh stored as 2xN array where N is the number of edges.
        The first row characterizes
    distances : np.ndarray
        distances of the edges in the graph.
    direct_neigh_idxs : np.ndarray
        Array that containes the direct neigbours of a vertex
    n_edges : np.ndarray
        Number of adjacent vertices per vertex
    k : int
        Number of neighbours that should be found

    Returns
    -------
    fedges : np.ndarray
        An array of size 3xk.
        The first two rows are the neighbourhood connections in a source,
        target format. The last row stores the distances between the
        nearest neighbours.

    """
    # output structures
    fedges = np.zeros((3, k), dtype=np.float32)
    fedges[0, :] = vi

    # a list of tuples where each tuple consist of a path and its length
    #shortest_paths = []
    paths_to_check = [(vi, 0)]
    # all paths that we observe
    #paths = []
    # bound to consider a path as nearest neighbour
    bound = sys.maxsize
    # does the shortest paths contain k neighbours?, i.e. len(sortest_paths) == k
    k_reached = False
    # dictionary containing all target vertices with the path length as value
    all_p_lens = {vi:0}
    # outer while loop
    while len(paths_to_check) > 0:
        #print("iter")
        # ---------BFS--------------
        tmp_paths_to_check = {}
        # we empty all paths to check at each iteration and fill them up at the end of the outer while loop
        while len(paths_to_check) > 0:
            target, path_distance = paths_to_check.pop(0)
            # if path is too long, we do not need to consider it anymore
            #if path_distance >= bound:
            #    continue
            # get the adjacent vertices of the target (last) vertex of this path 
            ns, ds = get_neigh(
                v=target,
                dv=path_distance,
                edges=edges,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                distances=distances)
            for z in range(ns.shape[0]):
                vn = int(ns[z])
                ds_z = ds[z]
                """
                ensure that you always save the shortest path to a target
                and that this shortest path is considered for future iterations
                """
                if vn in all_p_lens:
                    p_d = all_p_lens[vn]
                    if ds_z >= p_d:
                        continue
                all_p_lens[vn] = ds_z
                # new path that to be considered in the next iteration
                #tmp_paths_to_check.append((vn, ds_z))
                tmp_paths_to_check[vn] = ds_z
        # end inner while loop
        # sort the paths according to the distances in ascending order
        all_p_lens = dict(sorted(all_p_lens.items(), key=lambda x: x[1], reverse=False))
        # update the bound
        if len(all_p_lens) >= k+1:
            old_bound = bound
            bound = list(all_p_lens.values())[k]
            #if bound > old_bound:
            #    raise Exception("Bound Error: {0}, {1}".format(bound, old_bound))
        
        # throw paths away that have a larger distance than the bound
        """for j in range(len(tmp_paths_to_check)):
            target, path_distance = tmp_paths_to_check[j]
            if path_distance >= bound:
                continue
            paths_to_check.append((target, path_distance))"""
        for vert, dist in tmp_paths_to_check.items():
            if dist >= bound:
                continue
            paths_to_check.append((vert, dist))

    # end outer while loop
    # finally, return thek nearest targets and distances
    added = 0
    for key, value in all_p_lens.items():
        if key == vi:
            continue
        if added == k:
            break
        fedges[1, added] = key
        fedges[2, added] = value
        added += 1
    if added != k:
        raise Exception("Vertex {2}: Only found {0}/{1} neighbours".format(added, k, vi))
    return fedges


def geodesics(v_idx, direct_neigh_idxs, n_edges, edges, distances, target, node_distance, tmp_distances, tmp_neighbours, depth, depths, start_vertex, num_edges):
    depth += 1
    if len(tmp_distances) >= target:
        # if we have enough neighbours
        return tmp_distances, tmp_neighbours, depths
    else:
        # we have to consider more neighbours
        # query the neighbours of a point
        start_i = direct_neigh_idxs[v_idx]
        stop_i = start_i + n_edges[v_idx]
        # direct neighbours of the vertex 'v_idx'
        neighbour_vertices = edges[1, start_i:stop_i]
        # the distances of the neighbours are the distance to this node/vertex plus the distances to the neighbourhood nodes
        neigh_dists = node_distance + distances[start_i:stop_i]

        # vertex idxs of the neighbourhood that should be considered
        neighbour_vertices_ = []
        # distances of that neighbours to the vertex 'v_idx'
        neigh_dists_ = []
        # filter vertices that are already in the neighbourhood of the vertex 'start_vertex'
        for j in range(neighbour_vertices.shape[0]):
            neighbour_v_idx = neighbour_vertices[j]
            neigh_dist = neigh_dists[j]
            if neighbour_v_idx in tmp_neighbours:
                continue # this vertex is already in our neighbourhood 
            elif neighbour_v_idx == start_vertex:
                continue # this vertex is the start vertex
            # we definitely should consider this vertex
            neighbour_vertices_.append(neighbour_v_idx)
            neigh_dists_.append(neigh_dist)
        if len(neigh_dists_) == 0:
            # we already considered the whole neighbourhood of this vertex
            return tmp_distances, tmp_neighbours, depths

        # extend the list of neighbour vertices
        tmp_neighbours.extend(neighbour_vertices_)
        tmp_distances.extend(neigh_dists_)
        depths.extend(len(neigh_dists_)*[depth])
        
        # lets check again if we have considered enough neighbours
        if len(tmp_distances) >= target:
            return tmp_distances, tmp_neighbours, depths

        # check more neighbourhood at deeper depth
        for j in range(len(neigh_dists_)):
            # vertex idx of this neighbour
            neighbour_v_idx = neighbour_vertices_[j]
            # distance from the parent vertex to this neighbour
            dist = neigh_dists_[j]
            tmp_distances, tmp_neighbours, depths = geodesics(
                v_idx=neighbour_v_idx,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                edges=edges,
                distances=distances,
                target=target,
                node_distance=node_distance+dist,
                tmp_distances=tmp_distances,
                tmp_neighbours=tmp_neighbours,
                depth=depth,
                depths=depths,
                start_vertex=start_vertex, 
                num_edges=num_edges)
        # we considered all neighbours of that vertex
        return tmp_distances, tmp_neighbours, depths


def mesh_edges_distances(mesh_vertices, mesh_tris, adj_list, knn=45, respect_direct_neigh=False, 
        use_cartesian=False, bidirectional=False, n_proc=1, ignore_knn=False, verbose=True):
    """The function calculates a k-nearest neigbour (knn) graph over the surface of a mesh. 

    Parameters
    ----------
    mesh_vertices : np.ndarray
        Vertices of the mesh
    mesh_tris : np.ndarray
        Triangles of the mesh as triples of vertex indices
    adj_list : list()
        A list of neighbour vertex indices per vertex
    knn : int
        Number of neigbours that should be found
    use_cartesian : bool
        Use the cartesian product to link all superpoints.
    bidirectional : bool
        Make to nearest neighbour edges bidirectional.
    respect_direct_neigh : bool
        Keep the direct neighbours into the neighbourhood.
    n_proc : int
        How many processes should be used to calculate the nearest neighbours and superpoint features.

    Returns
    -------
    dict(source : np.ndarray, target : np.ndarray, distances : np.ndarray)
        Source and target are the edges in the knn-graph. The distances between
        two vertices in the knn-graph are stored in the distances array.
    """
    if use_cartesian:
        # first, we calculate the edges of the mesh via a cartesian product
        # the edges will be stored in this list 
        edges = []
        for al in adj_list:
            cartesian = list(itertools.product(*[al, al]))
            n = len(al)

            # delete self loop of the vertices
            r = np.arange(n)
            idxs_to_del = n * r + r
            for idx in reversed(idxs_to_del):
                del cartesian[int(idx)]
            # cartesian are the direct neighbours of a point P
            edges.extend(cartesian)
        # edges as 2Xn array
        edges = np.array(edges, dtype=np.uint32).transpose()
        edges = np.unique(edges, axis=1)
        # sort the source vertices
        sortation = np.argsort(edges[0])
        edges = edges[:, sortation]
    else:
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
        if bidirectional:
            r_edges = np.zeros((2, edges.shape[1]), dtype=np.uint32)
            r_edges[0, :] = edges[1]
            r_edges[1, :] = edges[0]
            edges = np.hstack((edges, r_edges))
        
        # sort the source vertices
        sortation = np.argsort(edges[0])
        edges = edges[:, sortation]
        #print(edges[:, :100])
    
    # edges in source, target layout
    source = edges[0]
    target = edges[1]

    # distances of the edges from a vertex v_i to v_j 
    distances = np.sqrt(np.sum((mesh_vertices[source] - mesh_vertices[target])**2, axis=1))
    
    if ignore_knn:
        return {"source": source, "target": target, "distances": distances}

    # unique vertices with the begin of each vertex (which is stored in 'direct_neigh_idxs') plus the number of neighbours 'n_edges'
    uni_verts, direct_neigh_idxs, n_edges = np.unique(edges[0, :], return_index=True, return_counts=True)
    max_idx = int(uni_verts.max())
    if uni_verts.shape[0] != (max_idx + 1):
        raise Exception("Max idx {0} with shape {1}".format(max_idx, uni_verts.shape[0]))
    # target number of edges
    f_target = knn*mesh_vertices.shape[0]
    # container where we store the final edges
    f_edges = np.zeros((2, f_target), dtype=np.uint32)
    f_distances = np.zeros((f_target, ), dtype=np.float32)

    # number of original edges
    num_edges = edges.shape[1]
    # kd tree is used if the geodesic neighbourhood is not large enough (< knn) which is a rare case
    tree = KDTree(data=mesh_vertices)
    
    if respect_direct_neigh:
        # check if direct neighbourhood is greater than 'knn'
        vi = 0
        # vertex indices that have a smaller neighbourhood than 'knn'
        idxs_todo = []
        for v_idx in tqdm(range(uni_verts.shape[0]), desc="Sampling"):
            # number of edges of this vertex 'v_idx'
            n_edges_vertex = n_edges[v_idx]
            # set the source vertex for every edge as this vertex 'v_idx'
            i_stop = vi+knn      
            f_edges[0, vi:i_stop] = v_idx
            if n_edges_vertex > knn:
                # we have more edges than knn -> sample the knn nearest neighbours

                # determine the interval of the neighbourhood for the edges array
                start_i = direct_neigh_idxs[v_idx]
                stop_i = start_i + n_edges_vertex
                
                # extract the neighbourhood with the distances of vertex 'v_idx'
                neighbour_vertices = edges[1, start_i:stop_i]
                neighbour_distances = distances[start_i:stop_i]
                
                # sort the neighbours and the distances according to the distances
                sortation = np.argsort(neighbour_distances)
                neighbour_vertices = neighbour_vertices[sortation]
                neighbour_distances = neighbour_distances[sortation]
                
                # sample the knn nearest neighbours
                neighbour_vertices = neighbour_vertices[:knn]
                neighbour_distances = neighbour_distances[:knn]

                # save the nearest neighbours in the final arrays f_edges and f_distances
                f_edges[1, vi:i_stop] = neighbour_vertices
                f_distances[vi:i_stop] = neighbour_distances
            else:
                # we have less or equal edges than knn
                
                # determine the interval of the neighbourhood of this vertex 'v_idx'
                n_neighbours = n_edges[v_idx]
                start_i = direct_neigh_idxs[v_idx]
                stop_i = start_i + n_neighbours
                
                # assign the direct neighbourhood to this vertex
                i_stop = vi + n_neighbours
                f_edges[1, vi:i_stop] = edges[1, start_i:stop_i]
                f_distances[vi:i_stop] = distances[start_i:stop_i]
                
                if n_edges_vertex < knn:
                    # we have to search the nearest neighbours with resprect to the surface (next for loop)
                    idxs_todo.append(v_idx)
            vi += knn

        # for every vertex where the geodesic search has to be conducted
        for k in tqdm(range(len(idxs_todo)), desc="Calculate geodesics"):
            # extract the vertex index
            v_idx = idxs_todo[k]
            # extract the number of edges of that vertex 'v_idx'
            n_edges_vertex = n_edges[v_idx]
            # output lists
            tmp_distances = []
            tmp_neighbours = []
            depths = []
            depth = 0
            tmp_distances, tmp_neighbours, depths = geodesics(
                v_idx=v_idx,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                edges=edges,
                distances=distances,
                target=knn,
                node_distance=0,
                tmp_distances=tmp_distances,
                tmp_neighbours=tmp_neighbours,
                depth=depth,
                depths=depths,
                start_vertex=v_idx,
                num_edges=num_edges)
            if len(tmp_neighbours) < knn:
                # we have found less neighbours than knn
                missing = knn - len(tmp_neighbours)
                # search the euclidean nearest neighbours which we haven't considered as neighbours yet 
                v = mesh_vertices[v_idx]
                nn_distances, nn_indices = tree.query(x=v, k=knn+1)
                nn_distances = nn_distances[1:]
                nn_indices = nn_indices[1:]
                # filter vertices that we already consider as neighbours
                add_n = []
                add_d = []
                for j in range(len(nn_indices)):
                    idx = nn_indices[j]
                    if idx in tmp_neighbours:
                        # we already consider this vertex as neighbour
                        continue
                    add_n.append(idx)
                    add_d.append(nn_distances[j])
                    # check if we have enough neighbours
                    if len(add_n) == missing:
                        break
                if missing == 1:
                    # we only need to set one neighbour
                    # determine the index of that neighbour
                    vi = knn * v_idx + knn - 1
                    # save the edge and the distance
                    f_edges[0, vi] = v_idx
                    f_edges[1, vi] = add_n[0]
                    f_distances[vi] = add_d[0]
                else:
                    # we need to set the 'missing neighbours'
                    # determine the interval where have to set the neighbours
                    vi_stop = knn * v_idx + knn
                    vi_start = vi_stop - missing
                    # save the edges and the distances
                    f_edges[0, vi_start:vi_stop] = v_idx
                    f_edges[1, vi_start:vi_stop] = add_n
                    f_distances[vi_start:vi_stop] = add_d
            else:
                # we need to filter the neighbours as there are more neighbours than knn
                # number of neighbour that we need to set
                residual = knn - n_edges_vertex

                # exclude the direct neighbourhood to only consider vertices with depth >= 2
                tmp_neighbours = np.array(tmp_neighbours[n_edges_vertex:])
                tmp_distances = np.array(tmp_distances[n_edges_vertex:])

                # sort them according to their distances
                sortation = np.argsort(tmp_distances)
                tmp_neighbours = tmp_neighbours[sortation]
                tmp_distances = tmp_distances[sortation]
                
                # query the necessary amount of  neighbours and distances
                tmp_neighbours = tmp_neighbours[:residual]
                tmp_distances = tmp_distances[:residual]

                # determine the interval to set the neighbours and their distances
                vi_stop = knn * v_idx + knn
                vi_start = vi_stop - residual
                if residual == 1:
                    # we only need one neighbour
                    f_edges[0, vi_start] = v_idx
                    f_edges[1, vi_start] = tmp_neighbours[0]
                    f_distances[vi_start] = tmp_distances[0]
                else:
                    f_edges[0, vi_start:vi_stop] = v_idx
                    f_edges[1, vi_start:vi_stop] = tmp_neighbours
                    f_distances[vi_start:vi_stop] = tmp_distances
    else:
        if n_proc > 1:
            args = [(vidx, edges, distances, direct_neigh_idxs, n_edges, knn) for vidx in range(uni_verts.shape[0])]
            t1 = time.time()
            print("Use {0} processes".format(n_proc))
            with Pool(processes=n_proc) as p:
                res = p.starmap(search_bfs, args)
            t2 = time.time()
            medges = np.hstack(res)
            f_edges = medges[:2, :].astype(np.uint32)
            f_distances = medges[2, :]
            print("Done in {0:.3f} seconds".format(t2-t1))
        else:
            """
            for s in tqdm(range(uni_verts.shape[0]), desc="Check Bidirectionality"):
                start = direct_neigh_idxs[s]
                stop = start + n_edges[s]
                neighbours = edges[1, start:stop]
                for j in range(neighbours.shape[0]):
                    t = neighbours[j]
                    startt = direct_neigh_idxs[t]
                    stopt = startt + n_edges[t]
                    bidirec = s in edges[1, startt:stopt]
                    if not bidirec:
                        print(s, t)
                        raise Exception("not bidirec")
            """

            vi = 0
            f_edges = np.zeros((2, uni_verts.shape[0]*knn), dtype=np.uint32)
            f_distances = np.zeros((uni_verts.shape[0]*knn, ), dtype=np.float32)
            arr_idx = 0
            for v_idx in tqdm(range(uni_verts.shape[0]), desc="Searching", disable=not verbose):
                fedges = search_bfs(vi=v_idx, edges=edges, distances=distances,
                    direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges, k=knn)
                f_edges[0, arr_idx:arr_idx+knn] = v_idx
                f_edges[1, arr_idx:arr_idx+knn] = fedges[1, :]
                f_distances[arr_idx:arr_idx+knn] = fedges[2, :]
                arr_idx += knn
    source = f_edges[0]
    target = f_edges[1]

    return {"source": source, "target": target, "distances": f_distances}


def calculate_stris(tris, partition_vec, sp_idxs, verbose=True):
    """
    Calculate a partition of triangles, the so called supertriangles, and the 
    superedges that connect two superpoints.

    Parameters
    ----------
    tris : np.ndarray 
        Array with triangles as three vertex indices (Size: #tris x 3).
    partition_vec : np.ndarray
        Vector that represents the partition.
    sp_idxs : list(np.ndarray)
        A list of superpoints. A superpoint consist of a set of vertex indices.

    Returns
    -------
    stris : np.ndarray
        A list of supertriangles. A supertrianlge is a set of triangles.
    v_to_move : list(tuple(idx0, idx1, idx2, sp_idx))
        A list of tuples. A tuple consist of three vertex indices and a superpoint
        index where the vertices belong to. 
    ssizes : list(int)
        A list with the size of each superpoint
    sedges : list()
        A list of links between the superpoints
    """
    sedges = []
    stris = []
    ssizes = len(sp_idxs) * [None]
    for i in range(len(sp_idxs)):
        stris.append(list())
        ssizes[i] = len(sp_idxs[i])
    v_to_move = []
    for i in tqdm(range(tris.shape[0]), desc="Determine Supertriangles and -edges", disable=not verbose):
        tri = tris[i]

        # determine the vertices of a triangle
        idx0 = tri[0]
        idx1 = tri[1]
        idx2 = tri[2]
        
        # determine the superpoint index of each vertex in the triangle
        sp0 = partition_vec[idx0]
        sp1 = partition_vec[idx1]
        sp2 = partition_vec[idx2]

        # determine if the triangles is shared by at least two superpoints
        is_not_shared = True
        if sp0 != sp1:
            # is shared
            is_not_shared = False
            sedges.append([sp0, sp1])
        if sp0 != sp2:
            is_not_shared = False
            sedges.append([sp0, sp2])
        if sp1 != sp2:
            is_not_shared = False
            sedges.append([sp1, sp2])
        if is_not_shared:
            # we can take any superpoint index and assign the triangle to this index
            stris[sp0].append(i)
        else: 
            # the largest superpoint will get the triangle and the vertices 
            s0 = ssizes[sp0]
            s1 = ssizes[sp1]
            s2 = ssizes[sp2]
            idx = np.argmax([s0, s1, s2])
            sps = [sp0, sp1, sp2]
            # index of the largest superpoint
            big_sp = sps[idx]
            stris[big_sp].append(i)
            # all points must now move to big_sp
            v_to_move.append((idx0, idx1, idx2, big_sp))
    # store the final triangle partition as numpy arrays
    for i in range(len(stris)):
        stris[i] = np.array(stris[i], dtype=np.uint32)
    return stris, v_to_move, ssizes, sedges


def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
 
        mid = (high + low) // 2
 
        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        # Element is not present in the array
        return -1


def clean_edges_helper(i, uni_source, uni_index, uni_counts, target):
    start = uni_index[i]
    stop = start + uni_counts[i]
    u_s = uni_source[i]
    res = []
    for j in range(start, stop):
        res.append([i, j, -1])
        u_t = target[j]
        if u_t > u_s: # search right
            # search index of u_t in the source array
            t_idx = binary_search(arr=uni_source, low=i, high=uni_source.shape[0], x=u_t)
        else: # search left
            t_idx = binary_search(arr=uni_source, low=0, high=i, x=u_t)
        if t_idx == -1:
            raise Exception("Index not found")

        t_start = uni_index[t_idx]
        t_stop = t_start + uni_counts[t_idx]
        r_idx = -1
        for r_i in range(t_start, t_stop): # iterate through source/target idxs of u_t
            if target[r_i] == u_s: # find the target u_s, i.e. the reverse edge (u_t, u_s) according to the edge (u_s, u_t)
                r_idx = r_i # we do not need this edge anymore
                break
        if r_idx == -1:
            continue
        res[-1][-1] = r_idx
    return res


def clean_edges_threads(d_mesh, verbose=False):
    t1 = time.time()
    source = d_mesh["source"]
    target = d_mesh["target"]
    distances = d_mesh["distances"]

    uni_source, uni_index, uni_counts = np.unique(source, return_index=True, return_counts=True)
    mask = np.ones((source.shape[0], ), dtype=np.bool)
    checked = np.zeros((source.shape[0], ), dtype=np.bool)
    if verbose:
        print("start threads")
        print("process {0} unique edges".format(uni_source.shape[0]))
    #"""
    tpool = ThreadPool(1000)
    w_args = []
    #for i in range(1000):
    for i in range(uni_source.shape[0]):
        #w_args.append((i, np.array(uni_source, copy=True), np.array(uni_index, copy=True), np.array(uni_counts, copy=True), np.array(target, copy=True)))
        w_args.append((i, uni_source, uni_index, uni_counts, target))

    results = tpool.starmap(clean_edges_helper, w_args)
    #"""
    t2 = time.time()
    if verbose:
        print("threads finished in {0:.3f} seconds".format(t2-t1))
    #print(results)

    t1 = time.time()
    for i in range(len(results)):
        result = results[i]
        for j in range(len(result)):
            res = result[j]
            r_idx = res[2] # the reverse index in the source, target array
            ui = res[0] # source idx in uni_source
            uj = res[1] # target idx in source, target array
            if checked[ui]:
                continue
            if r_idx == -1:
                continue
            checked[uj] = True
            mask[r_idx] = False
    t2 = time.time()
    c_source = np.array(source[mask], copy=True)
    c_target = np.array(target[mask], copy=True)
    c_distances = np.array(distances[mask], copy=True)
    if verbose:
        print("post processed edges in {0:.3f} seconds ".format(t2-t1))
    return {
        "source": source,
        "target": target,
        "c_source": c_source,
        "c_target": c_target,
        "distances": distances,
        "c_distances": c_distances
    }



def clean_edges(d_mesh, verbose=False):
    source = d_mesh["source"]
    target = d_mesh["target"]
    distances = d_mesh["distances"]

    uni_source, uni_index, uni_counts = np.unique(source, return_index=True, return_counts=True)
    mask = np.ones((source.shape[0], ), dtype=np.bool)
    checked = np.zeros((uni_source.shape[0], ), dtype=np.bool)

    for i in tqdm(range(uni_source.shape[0]), desc="Remove bidirectional edges", disable=not verbose):
        start = uni_index[i]
        stop = start + uni_counts[i]
        u_s = uni_source[i]
        for j in range(start, stop):
            u_t = target[j]
            t_idx = np.where(uni_source == u_t)[0][0]
            if checked[t_idx]:
                #print("check")
                continue
            t_start = uni_index[t_idx]
            t_stop = t_start + uni_counts[t_idx]
            #print(t_idx, t_start, t_stop)
            reverse_idxs = np.where(target[t_start:t_stop] == u_s)[0]
            if reverse_idxs.shape[0] == 0:
                continue
            reverse_idxs += t_start
            mask[reverse_idxs] = False
        checked[i] = True
        
    #print("Delete {0} reverse edges".format(np.sum(mask)))
    c_source = np.array(source[mask], copy=True)
    c_target = np.array(target[mask], copy=True)
    c_distances = np.array(distances[mask], copy=True)

    return {
        "source": source,
        "target": target,
        "c_source": c_source,
        "c_target": c_target,
        "distances": distances,
        "c_distances": c_distances
    }


def get_d_mesh(xyz, tris, adj_list_, k_nn_adj, respect_direct_neigh, use_cartesian, bidirectional,
        n_proc, ignore_knn, verbose, g_dir, g_filename):
    try:
        #if ignore_knn:
        #    raise Exception()
        d_mesh = load_nn_file(fdir=g_dir, fname=g_filename, a=k_nn_adj, verbose=verbose, ignore_knn=ignore_knn)
    except:
        if verbose:
            print("Calculate geodesic nearest neighbours, {0} vertices, {1} triangles".format(xyz.shape[0], tris.shape[0]))
        d_mesh = mesh_edges_distances(
            mesh_vertices=xyz,
            mesh_tris=tris,
            adj_list=adj_list_,
            knn=k_nn_adj,
            respect_direct_neigh=respect_direct_neigh,
            use_cartesian=use_cartesian,
            bidirectional=bidirectional,
            n_proc=n_proc,
            ignore_knn=ignore_knn,
            verbose=verbose)
        #d_mesh = clean_edges(d_mesh=d_mesh)
        d_mesh = clean_edges_threads(d_mesh=d_mesh)
        try:
            #if ignore_knn:
            #    raise Exception()
            save_nn_file(fdir=g_dir, fname=g_filename, d_mesh=d_mesh, a=k_nn_adj, verbose=verbose, ignore_knn=ignore_knn)
        except Exception as e:
            if verbose:
                print("Nearest neighbours not saved")
        if verbose:
            print("Done")
    return d_mesh


def superpoint_graph_mesh(mesh_vertices_xyz, mesh_vertices_rgb, mesh_tris, adj_list, 
        lambda_edge_weight=1, reg_strength=0.1, d_se_max=0, k_nn_adj=45, use_cartesian=True, 
        bidirectional=False, respect_direct_neigh=False, n_proc=1, move_vertices=False,
        g_dir=None, g_filename=None, ignore_knn=False, verbose=True, smooth=True, with_graph_stats=False):
    """Partitions a mesh.

    Parameters
    ----------
    mesh_vertices_xyz : o3d.geometry.Vector3dVector
        Vertices of the mesh.
    mesh_vertices_rgb : o3d.geometry.Vector3dVector
        Vertex colours of the mesh.
    mesh_tris : o3d.geometry.Vector3iVector
        Triangles of the mesh.
    adj_list : list()
        Adjacancy list for each point, i.e. the adjacent points of each point.
    lambda_edge_weight : float
        Parameter determine the edge weight for minimal part.
    reg_strength : float
        Regularization strength for the minimal partition. Increase lambda for a coarser partition.
    d_se_max : float
        Can be used to ignore long edges in the nearest neighbour graph.
    k_nn_adj : int
        Number of neighbours of the nearest neighbour graph.
    use_cartesian : bool
        Use the cartesian product to link all superpoints.
    bidirectional : bool
        Make to nearest neighbour edges bidirectional.
    respect_direct_neigh : bool
        Keep the direct neighbours into the neighbourhood.
    n_proc : int
        How many processes should be used to calculate the nearest neighbours and superpoint features.
    move_vertices : bool
        If True, then the vertices that are in triangles that are in different
        superpoints will belong to the biggest of these superpoint. 
    g_dir : str
        Directory to store the temporary files.
    g_filename : str
        Name of the temporary file
    ignore_knn : bool
        Use the triangular structure of the mesh to compute the features which are used as input for the l0-CP
    smooth : bool
        Use geodesic nearest neighbour features as input for the l0-CP

    Returns
    -------
    n_com : int
        Number of superpoints.
    n_sedg : int
        Number of superedges.
    components : list(np.ndarray)
        The list of superpoints. A superpoint consist of point indices.
    senders : np.ndarray
        The sender nodes of an edge.
    receivers : np.ndarray
        The receiver nodes of an edge.
    stris : list(np.ndarray)
        A partition of the triangles.
    """
    adj_list_ = adj_list.copy()
    xyz = np.array(np.asarray(mesh_vertices_xyz), copy=True)
    rgb = np.array(np.asarray(mesh_vertices_rgb), copy=True)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    tris = np.array(np.asarray(mesh_tris), copy=True)

    d_mesh = get_d_mesh(xyz=xyz, tris=tris, adj_list_=adj_list_, k_nn_adj=k_nn_adj, respect_direct_neigh=respect_direct_neigh,
        use_cartesian=use_cartesian, bidirectional=bidirectional, n_proc=n_proc, ignore_knn=ignore_knn, verbose=verbose,
        g_dir=g_dir, g_filename=g_filename)

    # TODO we could add geodesic features that leverage the mesh structure
    if ignore_knn:
        if smooth:
            d_mesh_ = get_d_mesh(xyz=xyz, tris=tris, adj_list_=adj_list_, k_nn_adj=k_nn_adj, respect_direct_neigh=respect_direct_neigh,
                use_cartesian=use_cartesian, bidirectional=bidirectional, n_proc=n_proc, ignore_knn=False, verbose=verbose,
                g_dir=g_dir, g_filename=g_filename)
            geof = libply_c.compute_geof(xyz, d_mesh_["target"], k_nn_adj, False).astype(np.float32)
        else:
            n_per_v = np.zeros((len(adj_list_), ), np.uint32)
            for i in range(len(adj_list_)):
                al = adj_list_[i]
                #print(i, al)
                n_per_v[i] = len(al)
            geof = libply_c.compute_geof_var(xyz, d_mesh["target"], n_per_v, False).astype(np.float32)
    else:
        geof = libply_c.compute_geof(xyz, d_mesh["target"], k_nn_adj, False).astype(np.float32)
    #choose here which features to use for the partition
    features = np.hstack((geof, rgb)).astype("float32")#add rgb as a feature for partitioning
    features[:,3] *= 2. #increase importance of verticality (heuristic)
    #print(features)
    verbosity_level = 0.0
    speed = 2.0
    #"""
    #"""
    if verbose:
        print("eIndex (python)", len(d_mesh["c_source"]))
    d_mesh["edge_weight"] = np.array(1. / ( lambda_edge_weight + d_mesh["c_distances"] / np.mean(d_mesh["c_distances"])), dtype = "float32")
    if verbose:
        print("Compute cut pursuit")

    components, in_component, stats = libcp.cutpursuit(features, d_mesh["c_source"], d_mesh["c_target"], d_mesh["edge_weight"], reg_strength, 0, 0, 1, verbosity_level, speed)
    stats = stats.tolist()
    stats[0] = int(stats[0]) # n ite main
    stats[1] = int(stats[1]) # iterations
    stats[2] = int(stats[2]) # exit code
    if with_graph_stats:
        geof = features[:, :4] # geometric features
        g_mean = np.mean(geof, axis=0)
        stats.extend(g_mean.tolist())
        g_std = np.std(geof, axis=0)
        stats.extend(g_std.tolist())
        g_med = np.median(geof, axis=0)
        stats.extend(g_med.tolist())
        w_mean = np.mean(d_mesh["edge_weight"])
        stats.append(w_mean)
        w_std = np.std(d_mesh["edge_weight"])
        stats.append(w_std)
        w_median = np.median(d_mesh["edge_weight"])
        stats.append(w_median)


    if verbose:
        print("Done")
        print("Python stats:", stats)
    # components represent the actual superpoints (partition) - now we need to compute the edges of the superpoints
    components = np.array(components, dtype = "object")
    # the number of components (superpoints) n_com
    n_com = max(in_component)+1

    if n_com == 1:
        return n_com, 0, components, None, None, None  

    in_component = np.array(in_component)  


    landrieu_method = False

    if landrieu_method:
        #interface select the edges between different superpoints
        interface = in_component[tris[:, 0]] != in_component[tris[:, 1]]
        n_inter_edges = interface.shape[0]
        #edgx and edgxr converts from tetrahedrons to edges (from i to j (edg1) and from j to i (edg1r))
        #done separatly for each edge of the tetrahedrons to limit memory impact
        edg1 = np.vstack((tris[interface, 0], tris[interface, 1]))
        edg1r = np.vstack((tris[interface, 1], tris[interface, 0]))
        
        interface = in_component[tris[:, 0]] != in_component[tris[:, 2]]
        n_inter_edges += interface.shape[0]
        # shape: (2, n_tetras)
        edg2 = np.vstack((tris[interface, 0], tris[interface, 2]))
        edg2r = np.vstack((tris[interface, 2], tris[interface, 0]))
        
        interface = in_component[tris[:, 1]] != in_component[tris[:, 2]]
        n_inter_edges += interface.shape[0]
        edg4 = np.vstack((tris[interface, 1], tris[interface, 2]))
        edg4r = np.vstack((tris[interface, 2], tris[interface, 1]))
        print("Number of edges that are between two superpoints: {0}/{1} ({2:.1f}%)".format(n_inter_edges, d_mesh["distances"].shape[0], 100*n_inter_edges/d_mesh["distances"].shape[0]))
        
        del interface
        # edges of points where one end point lies in a superpoint a und another end point in the superpoint b != a
        edges = np.hstack((edg1, edg2, edg4, edg1r, edg2r, edg4r))
        del edg1, edg2, edg4, edg1r, edg2r, edg4r  
        # Filter edges that occur multiple times in edges, e.g. [[x, x, z], [y, y, z]] -> [[x, z], [y, z]]
        edges = np.unique(edges, axis=1)
        if d_se_max > 0:
            # distance between the superpoints on the edges
            dist = np.sqrt(((xyz[edges[0,:]]-xyz[edges[1,:]])**2).sum(1))
            # filter edges with too large distance
            edges = edges[:,dist<d_se_max]
        
        #---sort edges by alpha numeric order wrt to the components (superpoints) of their source/target---
        # number of edges n_edg
        n_edg = len(edges[0])
        
        # replace point indices in edges with the superpoint idx of each point and save it in edge_comp
        edge_comp = in_component[edges]
        # len(edge_comp_index) == n_edg
        edge_comp_index = n_com * edge_comp[0,:] + edge_comp[1,:]
        order = np.argsort(edge_comp_index)
        
        # reordering of the edges, the edge_comp and the edge_comp_index array itself
        edges = edges[:, order]
        edge_comp = edge_comp[:, order]
        edge_comp_index = edge_comp_index[order]
        #print(edge_comp_index)

        """
        np.argwhere(np.diff(edge_comp_index)): where does the sorted edge_comp_index change? 
        This is stored in the array idxs_comp_change. The edge_comp array could have multiple 
        connections between the same superpoints. Therefore, we filter them with the jump_edg 
        in order to take once account of them. Thus, multiple edges from superpoint S_1 to S_2
        will be filtered so that we only have one edge from S_1 to S_2.  
        """
        idxs_comp_change = np.argwhere(np.diff(edge_comp_index)) + 1
        #marks where the edges change components iot compting them by blocks
        jump_edg = np.vstack((0, idxs_comp_change, n_edg)).flatten()
        n_sedg = len(jump_edg) - 1

        ########################################
        # Our modification
        ########################################
        senders = []
        receivers = []
        uni_edges = []
        print("Total number of superpoints (superedges): {0} ({1})".format(n_com, n_sedg))
        n_filtered = 0
        #---compute the superedges features---
        for i_sedg in tqdm(range(0, n_sedg), desc="Compute Superedges"):
            i_edg_begin = jump_edg[i_sedg]
            i_edg_end = jump_edg[i_sedg + 1]
            #ver_source = edges[0, range(i_edg_begin, i_edg_end)]
            #ver_target = edges[1, range(i_edg_begin, i_edg_end)]
            com_source = edge_comp[0, i_edg_begin]
            com_target = edge_comp[1, i_edg_begin]
            #xyz_source = xyz[ver_source, :]
            #xyz_target = xyz[ver_target, :]
            edge = (com_source, com_target)
            if edge in uni_edges:
                n_filtered += 1
                continue
            senders.append(com_source)
            receivers.append(com_target)
            inv_edge = (com_target, com_source)
            # bidirectional
            uni_edges.append(inv_edge)
    else:
        # calculate superedges and supertriangles
        stris, v_to_move, ssizes, sedges =\
            calculate_stris(
                tris=tris, partition_vec=in_component, sp_idxs=components, verbose=verbose)
        
        if move_vertices:
            # move vertices from superpoint A to B if it is part of a triangle that is inbetween 2 or more superpoints
            tmp_comps = np.array(components, copy=True)

            def move_v(comps, spX, sp, iX):
                idxs = comps[spX].tolist()
                idxs.remove(iX)
                comps[spX] = np.array(idxs, dtype=np.uint32)
                idxs = comps[sp].tolist()
                idxs.append(iX)
                comps[sp] = np.array(idxs, dtype=np.uint32)

            for i in range(len(v_to_move)):
                i0, i1, i2, sp = v_to_move[i]
                sp0 = in_component[i0]
                sp1 = in_component[i1]
                sp2 = in_component[i2]
                if sp0 != sp:
                    move_v(comps=tmp_comps, spX=sp0, sp=sp, iX=i0)
                if sp1 != sp:
                    move_v(comps=tmp_comps, spX=sp1, sp=sp, iX=i1)
                if sp2 != sp:
                    move_v(comps=tmp_comps, spX=sp2, sp=sp, iX=i2)
            components = tmp_comps

        # store the superedges
        sedges = np.array(sedges, dtype=np.uint32)
        if len(sedges) == 0:
            raise Exception("No superedges found")
        uni_edges = np.unique(sedges, axis=0)
        senders = uni_edges[:, 0].tolist()
        receivers = uni_edges[:, 1].tolist()
        n_sedg = uni_edges.shape[0]

    # make the superedges bidectional
    tmp_senders = senders.copy()
    senders.extend(receivers)
    receivers.extend(tmp_senders)
    senders = np.array(senders, dtype=np.uint32)
    receivers = np.array(receivers, dtype=np.uint32)
    return n_com, n_sedg, components, senders, receivers, stris, stats


def graph_mesh(mesh, reg_strength=0.1, lambda_edge_weight=1.0, k_nn_adj=30, use_cartesian=False, 
        bidirectional=False, respect_direct_neigh=False, n_proc=10, g_dir=None, g_filename=None,
        ignore_knn=False, smooth=True):
    """ This function creates a superpoint graph from a point cloud. 

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The input mesh.
    lambda_edge_weight : float
        Parameter determine the edge weight for minimal part.
    reg_strength : float
        Regularization strength for the minimal partition. Increase lambda for a coarser partition.
    k_nn_adj : int
        Number of neighbours of the nearest neighbour graph.
    use_cartesian : bool
        Use the cartesian product to link all superpoints.
    bidirectional : bool
        Make to nearest neighbour edges bidirectional.
    respect_direct_neigh : bool
        Keep the direct neighbours into the neighbourhood.
    n_proc : int
        How many processes should be used to calculate the nearest neighbours and superpoint features.
    g_dir : str
        Directory to store the temporary files.
    g_filename : str
        Name of the temporary file
    
    Returns
    -------
    dict : graph_dict
        The graph dictionary which is used by the neural network.
    sp_idxs : list[np.ndarray]
        A list of point indices for each superpoint.  
    stris : list[np.ndarray]
        A list of triangles per superpoint.
    P : np.ndarray
        The vertices of the mesh from which the features are calculated.
    n_sedg : int
        Number of superedges.
    """
    # make a copy of the original cloud to prevent that points do not change any properties
    cloud = np.hstack((np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors)))
    P = np.array(cloud, copy=True)
    print("Compute superpoint graph")
    t1 = time.time()

    mesh.compute_adjacency_list()

    n_sps, n_sedg, sp_idxs, senders, receivers, stris, _ = \
        superpoint_graph_mesh(
            mesh_vertices_xyz=mesh.vertices,
            mesh_vertices_rgb=mesh.vertex_colors,
            mesh_tris=mesh.triangles,
            adj_list=mesh.adjacency_list,
            reg_strength=reg_strength,
            lambda_edge_weight=lambda_edge_weight,
            k_nn_adj=k_nn_adj,
            use_cartesian=use_cartesian,
            bidirectional=bidirectional,
            respect_direct_neigh=respect_direct_neigh,
            n_proc=n_proc,
            g_dir=g_dir,
            g_filename=g_filename,
            move_vertices=False,
            ignore_knn=ignore_knn,
            smooth=smooth)

    """
    n_sps: Number of superpoints (vertices) in the graph
    n_edges: Number of edges in the graph
    sp_idxs: A list of point indices for each superpoint
    senders: List of start vertices for each edge in a directed graph
    receivers: List of end vertices for each edge in a directed graph
    """
    #############################################
    # create a feature vector for every superpoint
    t2 = time.time()
    duration = t2 - t1
    print("Superpoint graph has {0} nodes and {1} edges (duration: {2:.3f} seconds)".format(n_sps, n_sedg, duration))

    print("Compute features for every superpoint")

    t1 = time.time()
    P, center = feature_point_cloud(P=P)

    print("Check superpoint sizes")
    # so that every superpoint is smaller than the max_sp_size
    sps_sizes = []
    for i in range(n_sps):
        idxs = np.unique(sp_idxs[i])
        sp_idxs[i] = idxs
        #print(idxs)
        sp_size = idxs.shape[0]
        sps_sizes.append(sp_size)
    print("Average superpoint size: {0:.2f} ({1:.2f})".format(np.mean(sps_sizes), np.std(sps_sizes)))
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    P = np.hstack((P, normals))

    if n_sps == 1:
        n_sps = 2
        all_idxs = np.arange(P.shape[0])
        remaining_idxs = np.delete(all_idxs, sp_idxs[0])
        sp_idxs = sp_idxs.tolist()
        sp_idxs.append(remaining_idxs)
        sp_idxs = np.array(sp_idxs, dtype="object")

    args = [(k, sp_idxs, P) for k in range(n_sps)]
    print("Use {0} processes".format(n_proc))
    
    with Pool(processes=n_proc) as p:
        res = p.starmap(c, args)
    node_features = np.hstack(res)
    node_features = node_features.transpose()
    print("Nr of graph features {0}*{1}".format(node_features.shape[0], node_features.shape[1]))

    t2 = time.time()
    duration = t2 - t1
    print("Computed features in {0:.3f} seconds".format(duration))

    graph_dict = {
        "nodes": node_features,
        "senders": senders,
        "receivers": receivers,
        "edges": None,
        "globals": None
        }
    return graph_dict, sp_idxs, stris, P, n_sedg