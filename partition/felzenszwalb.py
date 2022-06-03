import numpy as np
import igraph as ig
import sys

def component_size(S, C):
    return np.where(S == C)[0].shape[0]

def tau(k, S, C):
    return k / component_size(S, C)

def internal_difference(G, S, C):
    # maximum edge weight that connects two nodes of the same component.
    E = G.get_edgelist()
    w_max = sys.float_info.min
    for k in range(len(E)):
        e = E[k]
        i, j = e
        if S[i] == C and S[j] == C:
            w = get_edge_weight(G, k)
            if w > w_max:
                w_max = w
    return w_max

def min_internal_difference(G, S, C1, C2, k):
    return min(internal_difference(G, S, C1) + tau(k, S, C1), internal_difference(G, S, C2) + tau(k, S, C2))

def merge(G, w, C1, C2, S, e, k):
    if w > min_internal_difference(G, S, C1, C2, k):
        return S
    i, j = e
    S[j] = S[i]
    return S

def sort_edges_ascending(G):
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
    S = np.arange(G.vcount(), dtype=np.uint32)
    return S

def get_n_edges(G):
    return len(G.get_edgelist())

def get_edge(G, idx):
    E = G.get_edgelist()
    return E[idx]

def get_edge_weight(G, idx):
    w = G.es["w"][idx]
    return w

def get_components_from_edge(S, e):
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

def partition_from_probs(graph_dict, sim_probs, k):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    all_verts = senders = np.vstack((senders[:, None], receivers[:, None]))
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
    S = partition(G, k)
    return S


if __name__ == "__main__":
    # similar nodes should have low edge weight
    G = ig.Graph()
    G.add_vertices(6)
    G.add_edges([(0,1), (1,2), (2, 3), (3, 4), (4, 5), (5, 3)])
    w = np.random.rand(G.ecount())
    G.es["w"] = w.tolist()
    p_vec = partition(G=G, k=100)
    print(p_vec)