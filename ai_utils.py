# cut pursuit library (see https://github.com/loicland/superpoint_graph/tree/ssp%2Bspg/partition)
import cp_ext as libcp
import ply_c_ext as libply_c

import numpy as np
import numpy.matlib
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from graph_nets import utils_tf
from tqdm import tqdm
import time

from network import FFGraphNet


def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0):
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
    print("Compute Graph NN")
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
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
    print("Done")
    return graph, target2


def superpoint_graph(xyz, rgb, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.1, d_se_max=0):
    """ This function is developed by Landrieu et al. 
    (see https://github.com/loicland/superpoint_graph). 
    We modified the output of the function to fit our use case. 
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    #---compute 10 nn graph-------
    # target_fea are the indices of the k nearest neighbours of each point (a point itself is not considered as neighbour)
    graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
    #---compute geometric features-------
    print("Compute geof")
    geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
    del target_fea

    #choose here which features to use for the partition
    features = np.hstack((geof, rgb/255.)).astype("float32")#add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
                
    graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = "float32")
    #print("        minimal partition...")
    print("Minimal Partition")
    """
    components: the actual superpoint idxs which is a list of lists (where each list contains point indices)
    in_component: this is one list with the length of number of points - here we got a superpoint idx for each point
        this is just another representation of components.
    """
    components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], reg_strength)
    print("Done")
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
    print("Total number of edges: {0}".format(n_sedg))
    n_filtered = 0
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
    print("{0} edges filtered, {1} unique edges".format(n_filtered, len(uni_edges)))
    return n_com, n_sedg, components, senders, receivers, len(uni_edges)


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


def init_model(n_ft):
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
    node_features = np.zeros((n_nodes, n_ft), dtype=np.float32)
    for i in range(n_nodes):
        sp = np.random.randn(100, 6)
        features = compute_features(cloud=sp)
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


def predict(graph_dict, dec_b=0.5):
    t1 = time.time()
    n_ft = graph_dict["nodes"].shape[1]
    model = init_model(n_ft=n_ft)
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