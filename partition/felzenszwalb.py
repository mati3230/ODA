import numpy as np
import igraph as ig
import sys

# TODO the size of the components, tau could be tracked while executing the algorithm
# TODO maximum edge weight of a component could also be tracked

def component_size(S, C):
    # returns the size of the component C (which is an integer number) in the partition S
    return np.where(S == C)[0].shape[0]

def tau(k, S, C):
    return k / component_size(S, C)

def max_weight(G, w, e_idxs):
    w_arr = np.array(w)
    return np.max(w_arr[e_idxs])

def internal_difference(G, S, C):
    # maximum edge weight w_max in the minimum spanning tree MST of the component C.
    if component_size(S=S, C=C) <= 1:
        return 0.0
    E = G.get_edgelist()
    G_C = ig.Graph()
    weights_C = []
    edges_C = []
    verts_C = {}
    v_idx = 0
    for k in range(len(E)):
        e = E[k]
        i, j = e
        if S[i] == C and S[j] == C:
            if i not in verts_C:
                verts_C[i] = v_idx
                v_idx += 1
            if j not in verts_C:
                verts_C[j] = v_idx
                v_idx += 1
            edges_C.append((verts_C[i],verts_C[j]))
            w = get_edge_weight(G, k)
            weights_C.append(w)
    if len(weights_C) == 0:
        return 0.0
    elif len(weights_C) == 1:
        return weights_C[0]
    elif len(weights_C) == 2:
        return min(weights_C)
    G_C.add_vertices(len(verts_C))
    G_C.add_edges(edges_C)
    edge_idxs_MST = G_C.spanning_tree(weights=weights_C, return_tree=False)
    w_max = max_weight(G=G_C, w=weights_C, e_idxs=edge_idxs_MST)
    return w_max

def min_internal_difference(G, S, C1, C2, k):
    # returns the minimum internal difference between the components C1 and C2
    return min(internal_difference(G, S, C1) + tau(k, S, C1), internal_difference(G, S, C2) + tau(k, S, C2))

def merge(G, w, C1, C2, S, e, k):
    # merge the components C1 and C2 if the edge weight w is greater than the minimum internal difference between them
    if w > min_internal_difference(G, S, C1, C2, k):
        return S
    i, j = e
    S[j] = S[i]
    return S

def sort_edges_ascending(G):
    # sort edges in ascending order w.r.t. the edge weights
    w = G.es["w"]
    w = np.array(w, dtype=np.float32)
    sortation = np.argsort(w)
    w_ = w[sortation]
    E = G.get_edgelist()
    E_ = []
    for i in range(len(E)):
        idx = sortation[i]
        e = E[idx]
        E_.append(e)
    G_ = ig.Graph()
    G_.add_vertices(G.vcount())
    G_.add_edges(E_)
    G_.es["w"] = w_.tolist()
    return G_

def initial_partition(G):
    # every vertex in the graph G is a component of the partition
    S = np.arange(G.vcount(), dtype=np.uint32)
    return S

def get_n_edges(G):
    # returns the number of edges in the graph G
    return len(G.get_edgelist())

def get_edge(G, idx):
    # returns an edge of the graph G by inserting the index idx
    E = G.get_edgelist()
    return E[idx]

def get_edge_weight(G, idx):
    # returns an edge weight of the graph G by inserting the index idx
    w = G.es["w"][idx]
    return w

def get_components_from_edge(S, e):
    # returns the components of vertices i and j that are part of the edge e=(i,j) 
    i, j = e
    return S[i], S[j]

def partition(G, k):
    G_ = sort_edges_ascending(G)
    S = initial_partition(G_)
    m = get_n_edges(G_)
    for i in range(m):
        e = get_edge(G_, i)
        w = get_edge_weight(G_, i)
        C1, C2 = get_components_from_edge(S, e)
        if C1 == C2:
            continue
        S = merge(G_, w, C1, C2, S, e, k)
    return S

def partition_from_probs(graph_dict, sim_probs, k, P, sp_idxs):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    all_verts = np.vstack((senders[:, None], receivers[:, None]))
    all_verts = np.unique(all_verts)
    n = all_verts.shape[0]
    G = ig.Graph()
    G.add_vertices(n)
    edges = []
    for i in range(senders.shape[0]):
        s = senders[i]
        r = receivers[i]
        e = (s, r)
        edges.append(e)
    G.add_edges(edges)
    G.es["w"] = 1 - sim_probs
    # now we now which superpoints belong together
    # S is a vector where each entry belongs to a superpoint
    # Merged superpoints have the same number as entry in S
    S = partition(G, k)

    uni_S = np.unique(S)
    S_ = np.zeros((P.shape[0], ), np.uint32)
    s_nr = 1
    for s in uni_S:
        s_idxs = np.where(S == s)[0]
        if s_idxs.shape[0] == 0:
            continue
        # superpoints in s_idxs belong together
        for sp_idx in s_idxs:
            # query their point indeces
            p_idxs = sp_idxs[sp_idx]
            
            S_[p_idxs] = s_nr
        s_nr += 1
    return S_


if __name__ == "__main__":
    # similar nodes should have low edge weight
    #G = ig.Graph()
    #G.add_vertices(6)
    #G.add_edges([(0,1), (1,2), (2, 3), (3, 4), (4, 5), (5, 3)])
    G = ig.Graph.Lattice([5, 5], circular=False)
    w = np.random.rand(G.ecount())    
    G.es["w"] = w.tolist()
    p_vec = partition(G=G, k=100)
    print(p_vec)