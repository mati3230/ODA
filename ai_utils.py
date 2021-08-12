import cp_ext as libcp
import ply_c_ext as libply_c
import numpy as np
import numpy.matlib
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from graph_nets import utils_tf

from network import FFGraphNet


def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0):
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
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
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    return graph, target2


def superpoint_graph(xyz, rgb, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.1, d_se_max=0):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    #---compute 10 nn graph-------
    graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
    #---compute geometric features-------
    geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
    del target_fea
    
    #choose here which features to use for the partition
    features = np.hstack((geof, rgb/255.)).astype("float32")#add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
                
    graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = "float32")
    #print("        minimal partition...")
    components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                 , graph_nn["edge_weight"], reg_strength)
    #print(components)
    #print(in_component)
    components = np.array(components, dtype = "object")
    n_com = max(in_component)+1

    in_component = np.array(in_component)

    tri = Delaunay(xyz)

    #interface select the edges between different components
    #edgx and edgxr converts from tetrahedrons to edges
    #done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
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
    edges = np.hstack((edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r ,edg5r, edg6r))
    del edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)
    
    if d_se_max > 0:
        dist = np.sqrt(((xyz[edges[0,:]]-xyz[edges[1,:]])**2).sum(1))
        edges = edges[:,dist<d_se_max]
    
    #---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0,:] +  edge_comp[1,:]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    #marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1

    senders = np.zeros((n_sedg, ), dtype='uint32')
    receivers = np.zeros((n_sedg, ), dtype='uint32')

    #---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        senders[i_sedg] = com_source
        receivers[i_sedg] = com_target

    return n_com, n_sedg, components, senders, receivers


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


def compute_features(P, n_curv=30, k_curv=14, k_far=30, n_normal=30, bins=10, min_r=-0.5, max_r=0.5):
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
    center = np.mean(P, axis=0)
    std = np.std(P, axis=0)
    median_color = np.median(P[:, 3:], axis=0)
    q_25_color = np.quantile(P[:, 3:], 0.25, axis=0)
    q_75_color = np.quantile(P[:, 3:], 0.75, axis=0)
    
    normals, curv = estimate_normals_curvature(P=P[:, :3], k_neighbours=k_curv)
    
    ########normals########
    # mean, std
    mean_normal = np.mean(normals, axis=0)
    std_normal = np.std(normals, axis=0)
    median_normal = np.median(normals, axis=0)
    q_25_normal = np.quantile(normals, 0.25, axis=0)
    q_75_normal = np.quantile(normals, 0.75, axis=0)

    # min, max
    ref_normal = np.zeros((normals.shape[0], 3))
    ref_normal[:, 0] = 1
    normal_dot, _, _ = np_fast_dot(a=ref_normal, b=normals)
    min_normal, max_normal, min_n_idxs, max_n_idxs = get_min_max(feature=normal_dot, target=n_normal)
    min_P_n, max_P_n = extract_min_max_points(P=P, min_idxs=min_n_idxs, max_idxs=max_n_idxs, target=n_normal, center=center, center_max=3)

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
    
    # min, max
    min_curv, max_curv, min_c_idxs, max_c_idxs = get_min_max(feature=curv, target=n_curv, pad=False)
    min_curv = 2 * (min_curv - 0.5)
    max_curv = 2 * (max_curv - 0.5) 
    target_c = int(n_curv / 2)
    min_curv = zero_padding(x=min_curv, target_size=target_c)
    max_curv = zero_padding(x=max_curv, target_size=target_c)
    min_P_c, max_P_c = extract_min_max_points(P=P, min_idxs=min_c_idxs, max_idxs=max_c_idxs, target=target_c, center=center, center_max=3)

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

    _, nns = get_knn(D=P[:, :3]) 

    k_nns = nns[:, 1:k_neighbours+1]
    p_nns = P[k_nns[:]]
    
    p = np.matlib.repmat(P, k_neighbours, 1)
    p = np.reshape(p, (n_P, k_neighbours, n))
    try:
        p = p - p_nns
    except Exception as e:
        normals = np.zeros((n_P, 3), dtype=np.float32)
        normals[:, 0] = 1
        curvature = np.zeros((n_P, ), dtype=np.float32)
        return normals, curvature
    
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

    return normals, curvature


def get_general_bb(P):
    bb = np.zeros((2*P.shape[1], ))
    j = 0
    for i in range(P.shape[1]):
        bb[j] = np.min(P[:, i])
        j += 1
        bb[j] = np.max(P[:, i])
        j += 1
    return bb


def get_knn(D):
    size = D.shape[0]
    dim = D.shape[1]
    # compute a distance matrix
    d_mat = np.zeros((size, size))
    for i in range(dim):
        # extract column
        col = D[:, i] 
        # convert vector to matrix
        col = col[:, None]
        # repeat column along rows
        A = np.tile(col, reps=[1, size])
        # copy and transpose
        A_T = np.array(A, copy=True).transpose()
        # calc intermediate distance matrix for column
        d_mat_tmp = A - A_T
        d_mat_tmp = np.square(d_mat_tmp)
        # add to final distance matrix
        d_mat += d_mat_tmp
        d_mat = np.sqrt(d_mat)
    n_idxs = np.argsort(d_mat, axis=1)
    return d_mat, n_idxs


def feature_point_cloud(P):
    center = np.mean(P[:, :3], axis=0)
    max_P = np.max(np.abs(P[:, :3]))
    P[:, :3] -= center
    P[:, :3] /= max_P
    P[:, 3:] /= 255
    P[:, 3:] -= 0.5
    P[:, 3:] *= 2
    return P, center


def graph(cloud, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.1, d_se_max=0, max_sp_size=7000):
    P = np.array(cloud, copy=True)
    print("Compute superpoint graph")
    n_sps, n_edges, sp_idxs, senders, receivers = superpoint_graph(
        xyz=P[:, :3],
        rgb=P[:, 3:],
        reg_strength=reg_strength,
        k_nn_geof=k_nn_geof,
        k_nn_adj=k_nn_adj,
        lambda_edge_weight=lambda_edge_weight,
        d_se_max=d_se_max)
    print("Superpoint graph has {0} nodes and {1} edges".format(n_sps, n_edges))
    print("Compute features for every superpoint")
    P, center = feature_point_cloud(P=P)

    sps_sizes = []
    idxs = np.unique(sp_idxs[0])
    sp_size = idxs.shape[0]
    if sp_size > max_sp_size:
        raise Exception("Superpoint {0} too large with {1} points (max: {2}). Try to lower the reg_strength.".format(0, sp_size, max_sp_size))
    sps_sizes.append(sp_size)
    sp = P[idxs]
    features = compute_features(P=sp)
    n_ft = features.shape[0]
    print("Use {0} features".format(n_ft))
    sp_idxs[0] = idxs
    node_features = np.zeros((n_sps, n_ft), dtype=np.float32)
    node_features[0] = features
    for k in range(1, n_sps):
        idxs = np.unique(sp_idxs[k])
        #print(idxs.shape)
        sp = P[idxs]
        sp_size = sp.shape[0]
        if sp_size > max_sp_size:
            raise Exception("Superpoint {0} too large with {1} points (max: {2}). Try to lower the reg_strength.".format(k, sp_size, max_sp_size))
        sps_sizes.append(sp_size)
        # TODO: remove random features!
        #features = np.random.randn(n_ft, )
        features = compute_features(P=sp)
        node_features[k] = features
        sp_idxs[k] = idxs
    print("Average superpoint size: {0:.2f} ({1:.2f})".format(np.mean(sps_sizes), np.std(sps_sizes)))

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
        features = compute_features(P=sp)
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
    n_ft = graph_dict["nodes"].shape[1]
    model = init_model(n_ft=n_ft)
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    
    a_dict = model.action(obs=input_graphs, training=False, decision_boundary=dec_b)
    
    action = a_dict["action"]
    action = action.numpy()
    action = action.astype(np.bool)
    
    probs = a_dict["probs"]
    probs = probs.numpy()

    return action, probs