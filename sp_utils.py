import numpy as np
from vedo import Plotter, Points, recoSurface, write, Mesh
import igraph as ig
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial.transform import Rotation


def extend(arr, n_arr, dtype):
    n_arr = np.array(n_arr, dtype=dtype)
    arr = np.vstack((arr[:, None], n_arr[:, None]))
    arr = arr.reshape(arr.shape[0], )
    return arr


def separate_superpoint(picked_points_idxs, sp_idxs, graph_dict, unions):
    if len(picked_points_idxs) != 1:
        print("Please choose only one point for separation. {0} chosen.".format(len(picked_points_idxs)))
        return graph_dict, unions
    sp_to_separate = points_to_superpoints(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)[0]
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    unions = mute_connections(sp_to_unify=[sp_to_separate], senders=senders, receivers=receivers, unions=unions)
    return graph_dict, unions


def find_idxs_by_val(a, idxs):
    idxs_to_del = np.zeros((0, 1), dtype=np.uint32)
    for i in range(idxs.shape[0]):
        idx = idxs[i]
        a_idxs = np.where(a == idx)[0]
        if a_idxs.shape[0] == 0:
            continue
        idxs_to_del = np.vstack((idxs_to_del, a_idxs[:, None]))
    idxs_to_del = idxs_to_del.reshape(idxs_to_del.shape[0], )
    idxs_to_del = np.unique(idxs_to_del)
    return idxs_to_del


def reduce_superpoint(picked_points_idxs, points_idxs_r, graph_dict, sp_idxs):
    if len(picked_points_idxs) != 1:
        print("Please choose one points for extending.")
        return graph_dict, sp_idxs
    if len(points_idxs_r) == 0:
        return graph_dict, sp_idxs
    source_sp = points_to_superpoints(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)[0]
    sps = points_to_superpoints(picked_points_idxs=points_idxs_r, sp_idxs=sp_idxs)
    sps = np.array(sps, dtype=np.uint32)
    point_idxs = np.array(points_idxs_r, dtype=np.uint32)
    sortation = np.argsort(sps)
    sps = sps[sortation]
    point_idxs = point_idxs[sortation]
    uni_sps, uni_idxs, uni_counts = np.unique(sps, return_index=True, return_counts=True)

    s_idx = np.where(uni_sps == source_sp)[0]
    if s_idx.shape[0] == 0:
        return graph_dict, sp_idxs
    s_idx = s_idx[0]

    uni_idx = uni_idxs[s_idx]
    uni_count = uni_counts[s_idx]
    start_i = uni_idx
    stop_i = start_i + uni_count
    
    P_idxs = sp_idxs[source_sp]

    P_idxs_to_del = point_idxs[start_i:stop_i]
    # remove target_point_idxs from target_sp
    del_idxs = find_idxs_by_val(a=P_idxs, idxs=P_idxs_to_del)
    #print(P_idxs.shape, P_idxs[del_idxs], P_idxs_to_del)
    P_idxs = np.delete(P_idxs, del_idxs)
    #print(P_idxs.shape)
    sp_idxs[source_sp] = P_idxs

    # add a new superpoint for this points
    sp_idxs = sp_idxs.tolist()
    sp_idxs.append(P_idxs_to_del)
    sp_idxs = np.array(sp_idxs, dtype="object")
    nodes = graph_dict["nodes"]
    new_node = np.zeros((1, nodes.shape[-1]), dtype=nodes.dtype)
    nodes = np.vstack((nodes, new_node))
    graph_dict["nodes"] = nodes
    return graph_dict, sp_idxs


def extend_superpoint_points(picked_points_idxs, points_idxs_e, sp_idxs):
    if len(picked_points_idxs) != 1:
        print("Please choose one points for extending.")
        return sp_idxs
    if len(points_idxs_e) == 0:
        return sp_idxs
    source_sp = points_to_superpoints(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)[0]
    sps = points_to_superpoints(picked_points_idxs=points_idxs_e, sp_idxs=sp_idxs)
    sps = np.array(sps, dtype=np.uint32)
    point_idxs = np.array(points_idxs_e, dtype=np.uint32)
    sortation = np.argsort(sps)
    sps = sps[sortation]
    point_idxs = point_idxs[sortation]
    uni_sps, uni_idxs, uni_counts = np.unique(sps, return_index=True, return_counts=True)

    for i in range(uni_counts.shape[0]):
        target_sp = uni_sps[i]
        if target_sp == source_sp:
            continue
        uni_idx = uni_idxs[i]
        uni_count = uni_counts[i]
        start_i = uni_idx
        stop_i = start_i + uni_count
        
        target_point_idxs = point_idxs[start_i:stop_i]
        
        t_P_idxs = sp_idxs[target_sp]
        change_idxs = find_idxs_by_val(a=target_point_idxs, idxs=t_P_idxs)
        t_P_idxs = np.delete(t_P_idxs, change_idxs)
        sp_idxs[target_sp] = t_P_idxs

        s_P_idxs = sp_idxs[source_sp]
        s_P_idxs = np.vstack((s_P_idxs[:, None], target_point_idxs[:, None]))
        s_P_idxs = s_P_idxs.reshape(s_P_idxs.shape[0], )
        sp_idxs[source_sp] = s_P_idxs
    return sp_idxs


def unify_superpoints(picked_points_idxs, sp_idxs, graph_dict, unions):
    if len(picked_points_idxs) != 2:
        print("Please choose only two points for union.")
    sps = points_to_superpoints(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
    sp_i = sps[0]
    sp_j = sps[1]
    if sp_i == sp_j:
        print("Choose different superpoints - nothing happened.")
        return unions
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]

    idxs = np.where((senders == sp_i) & (receivers == sp_j))[0]
    if idxs.shape[0] == 0:
        idxs = np.where((senders == sp_j) & (receivers == sp_i))[0]
    if idxs.shape[0] == 0:
        print("No edge available - nothing happened.")
        return unions
    if idxs.shape[0] != 1:
        raise Exception("Multi edge")
    unions[idxs] = True
    return unions


def extend_superpoint(picked_points_idxs, points_idxs_e, sp_idxs, graph_dict, unions):
    if len(picked_points_idxs) != 1:
        print("Please choose one points for extending.")
        return graph_dict, unions
    if len(points_idxs_e) != 1:
        print("Please choose one points for extending.")
        return graph_dict, unions
    source_sp = points_to_superpoints(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)[0]
    target_sp = points_to_superpoints(picked_points_idxs=points_idxs_e, sp_idxs=sp_idxs)[0]
    if source_sp == target_sp:
        print("2 points in the same superpoint.")
        return graph_dict, unions
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    unions = mute_connections(sp_to_unify=[target_sp], senders=senders, receivers=receivers, unions=unions)

    i_idxs = np.where(senders == target_sp)[0]
    j_idxs = np.where(receivers == source_sp)[0]
    edges_idxs = np.intersect1d(i_idxs, j_idxs)
    if len(edges_idxs) > 0:
        unions[edges_idxs] = True
        return graph_dict, unions
    i_idxs = np.where(receivers == target_sp)[0]
    j_idxs = np.where(senders == source_sp)[0]
    edges_idxs = np.intersect1d(i_idxs, j_idxs)
    if len(edges_idxs) > 0:
        unions[edges_idxs] = True
        return graph_dict, unions
    n_senders = []
    n_receivers = []
    n_senders.append(source_sp)
    n_receivers.append(target_sp)
    n_edges = 1
    #print("Add new edge from node {0} to {1}".format(source_sp, target_sp))
    senders, receivers, unions = extend_edges(
        senders=senders,
        n_senders=n_senders,
        receivers=receivers,
        n_receivers=n_receivers,
        n_edges=n_edges,
        unions=unions)
    graph_dict["senders"] = senders
    graph_dict["receivers"] = receivers
    
    return graph_dict, unions


def mute_connections(sp_to_unify, senders, receivers, unions):
    idxs_to_del = np.zeros((0, 1), dtype=np.uint32)
    for i in range(len(sp_to_unify)):
        sp_i = sp_to_unify[i]
        idxs = np.where(senders == sp_i)[0]
        idxs_to_del = np.vstack((idxs_to_del, idxs[:, None]))
        idxs = np.where(receivers == sp_i)[0]
        idxs_to_del = np.vstack((idxs_to_del, idxs[:, None]))
    idxs_to_del = idxs_to_del.reshape(idxs_to_del.shape[0], )
    idxs_to_del = np.unique(idxs_to_del)
    print("Remove {0} edges".format(idxs_to_del.shape[0]))
    unions[idxs_to_del] = False
    return unions


def extend_edges(senders, n_senders, receivers, n_receivers, n_edges, unions):
    senders = extend(arr=senders, n_arr=n_senders, dtype=np.uint32)
    receivers = extend(arr=receivers, n_arr=n_receivers, dtype=np.uint32)
    n_unions = np.ones((n_edges, 1), dtype=np.bool)
    unions = np.vstack((unions[:, None], n_unions))
    unions = unions.reshape(unions.shape[0], )
    return senders, receivers, unions


def unify(picked_points_idxs, sp_idxs, graph_dict, unions):
    if len(picked_points_idxs) == 0:
        return graph_dict, unions
    sp_to_unify = uni_superpoint_idxs(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
    print("Superpoints to unify: {0}".format(sp_to_unify))
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    n_sp_to_unify = len(sp_to_unify)

    unions = mute_connections(sp_to_unify=sp_to_unify, senders=senders, receivers=receivers, unions=unions)

    n_senders = []
    n_receivers = []
    n_edges = 0
    for i in range(n_sp_to_unify):
        sp_i = sp_to_unify[i]
        for j in range(i, n_sp_to_unify):
            sp_j = sp_to_unify[j]
            if sp_i == sp_j:
                continue
            i_idxs = np.where(senders == sp_i)[0]
            j_idxs = np.where(receivers == sp_j)[0]
            edges_idxs = np.intersect1d(i_idxs, j_idxs)
            if len(edges_idxs) > 0:
                unions[edges_idxs] = True
                continue
            i_idxs = np.where(receivers == sp_i)[0]
            j_idxs = np.where(senders == sp_j)[0]
            edges_idxs = np.intersect1d(i_idxs, j_idxs)
            if len(edges_idxs) > 0:
                unions[edges_idxs] = True
                continue
            n_senders.append(sp_i)
            n_receivers.append(sp_j)
            #print("Add new edge from node {0} to {1}".format(sp_i, sp_j))
            n_edges += 1
    if n_edges > 0:
        senders, receivers, unions = extend_edges(
            senders=senders,
            n_senders=n_senders,
            receivers=receivers,
            n_receivers=n_receivers,
            n_edges=n_edges,
            unions=unions)
    graph_dict["senders"] = senders
    graph_dict["receivers"] = receivers

    return graph_dict, unions


def points_to_superpoints(picked_points_idxs, sp_idxs):
    result = []
    for i in range(len(picked_points_idxs)):
        p_idx = picked_points_idxs[i]
        for j in range(len(sp_idxs)):
            idxs = sp_idxs[j]
            if p_idx in idxs:
                result.append(j)
                #print("Point idx {0} refers to superpoint {1}".format(p_idx, j))
                break
    return result


def uni_superpoint_idxs(picked_points_idxs, sp_idxs):
    result = points_to_superpoints(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
    result = np.unique(result)
    return result


def to_igraph(graph_dict, unions, half=False):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    if half:
        n_edges = int(senders.shape[0] / 2)
    else:
        n_edges = int(senders.shape[0])
    edge_list = n_edges*[None]
    for i in range(n_edges):
        vi = int(senders[i])
        vj = int(receivers[i])
        edge_list[i] = (vi, vj)
    g = ig.Graph(edges=edge_list)
    g.es["union"] = unions.tolist()
    #print("iGraph, n edges: {0}, n unions: {1}".format(len(edge_list), len(unions)))
    return g


def comp_list(graph_dict, unions, n_P, sp_idxs, half):
    ig_graph = to_igraph(graph_dict=graph_dict, unions=unions, half=half)
    unions = ig_graph.es["union"]
    unions = np.array(unions, dtype=np.bool)
    idxs = np.where(unions == False)[0]
    idxs = idxs.astype(np.uint32)
    idxs = idxs.tolist()
    ig_graph.delete_edges(idxs)
    components = ig_graph.clusters()
    components_list = len(components) * [None]
    for i in range(len(components)):
        comp = components[i]
        components_list[i] = len(comp)*[None]
        for j in range(len(comp)):
            sp_i = comp[j]
            P_idxs = sp_idxs[sp_i]
            components_list[i][j] = (sp_i, P_idxs)
    return components_list


def sp_in_comp(sp_i, comp):
    for i in range(len(comp)):
        sp_j = comp[i][0]
        if sp_i == sp_j:
            return True
    return False


def get_objects(picked_points_idxs, P, sp_idxs, graph_dict, unions):
    n_P = P.shape[0]
    selected_sps = uni_superpoint_idxs(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
    components_list = comp_list(graph_dict=graph_dict, unions=unions, n_P=n_P, sp_idxs=sp_idxs, half=False)
    n_comps = len(components_list)
    comps_to_reconstruct = []
    objects_to_reconstruct = []
    for i in range(selected_sps.shape[0]):
        sp_i = selected_sps[i]
        for j in range(n_comps):
            comp = components_list[j]
            if sp_in_comp(sp_i=sp_i, comp=comp):
                if j not in comps_to_reconstruct:
                    comps_to_reconstruct.append(j)
                    o_idxs = np.zeros((0, 1), dtype=np.uint32)
                    for k in range(len(comp)):
                        P_idxs = comp[k][1]
                        o_idxs = np.vstack((o_idxs, P_idxs[:, None]))
                    o_idxs = o_idxs.reshape(o_idxs.shape[0], )
                    objects_to_reconstruct.append(o_idxs)
                break
    return objects_to_reconstruct


def get_remaining(P, objects):
    n_P = P.shape[0]
    all_objs = np.zeros((0, 1), dtype=np.uint32)
    for i in range(len(objects)):
        obj = objects[i]
        all_objs = np.vstack((all_objs, obj[:, None]))
    all_objs = all_objs.reshape(all_objs.shape[0], )
    all_idxs = np.arange(n_P)
    remaining = np.delete(all_idxs, all_objs)
    return remaining


def to_reco_params(inp):
    inps = inp.split(",")
    if len(inps) != 3:
        return False, 150, 0.1, 15
    try:
        dims = int(inps[0])
        radius = float(inps[1])
        sample_size = int(inps[2])
    except Exception as e:
        print(e)
        return False, 150, 0.1, 15
    return True, dims, radius, sample_size


def reco(P, o_idxs, dims=150, radius=0.1, sample_size=15):
    #plt = Plotter(N=2, axes=0)
    pts0 = Points(P[o_idxs, :3], c=(255, 0, 0)).legend("cloud")
    #pts0 = pts0.clone().smoothMLS2D(f=0.8)  # smooth cloud
    #plt.show(pts0, "original point cloud", at=0)
    print("Reconstruct")
    reco = recoSurface(pts0, dims=dims, radius=radius, sampleSize=sample_size).legend("surf. reco")
    """
    reco.computeNormals()
    cldata = reco._data.GetCellData()
    tri_normals = None
    if cldata.GetNumberOfArrays():
        for i in range(cldata.GetNumberOfArrays()):
            iarr = cldata.GetArray(i)
            if iarr:
                icname = iarr.GetName()
                if icname == "Normals":
                    tri_normals = vtk_to_numpy(iarr)
    """
    # access
    # points: reco.points()
    # tris: reco.faces()
    # overview of the mesh: reco._data 
    print("Apply vertex colors")
    verts = np.array(reco.points())
    nn = NearestNeighbors(n_neighbors=1).fit(P[:, :3])
    distances, neighbors = nn.kneighbors(verts)
    P_n = P[neighbors[:, 0]]
    #print(neighbors[:, 0])
    #print(P_n)
    v_colors = P_n[:, 3:]
    tris = np.array(reco.faces())
    #mesh = Mesh([verts, tris])
    #mesh.color(c=v_colors)
    print("Set up o3d mesh")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors/255.)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    n_tris = tris.shape[0]
    print("Reconstruct with {0} tris".format(n_tris))
    print("Done")
    return mesh


def initial_partition(P, sp_idxs):
    if type(P) == o3d.geometry.TriangleMesh:
        vertices = np.asarray(P.vertices)
        n_P = vertices.shape[0]
    else:
        n_P = P.shape[0]

    par_v = np.zeros((n_P, ), dtype=np.uint32)
    n_sps = len(sp_idxs)
    print("Number of superpoints: {0}".format(n_sps))
    for i in range(n_sps):
        idxs = sp_idxs[i]
        par_v[idxs] = i+1
    return par_v


def partition(graph_dict, unions, P, sp_idxs, half=False, stris=None):
    is_mesh = False
    if type(P) == o3d.geometry.TriangleMesh:
        vertices = np.asarray(P.vertices)
        vertex_colors = np.asarray(P.vertex_colors)
        n_P = vertices.shape[0]
        meshes = []
        is_mesh = True
    else:
        n_P = P.shape[0]
    par_v = np.zeros((n_P, ), dtype=np.uint32)
    c_list = comp_list(graph_dict=graph_dict, unions=unions, n_P=n_P, sp_idxs=sp_idxs, half=half)

    if is_mesh:
        for i in range(len(c_list)):
            comp = c_list[i]
            n_sp_comp = len(comp)
            nstris = []
            for j in range(n_sp_comp):
                sp_idx = comp[j][0]
                P_idxs = comp[j][1]
                par_v[P_idxs] = i + 1
                stri = stris[sp_idx]
                nstris.append(stri)
            triangles = np.vstack(nstris)
            uni_t = np.unique(triangles)
            n_triangles = np.array(triangles, copy=True)
            for k in range(uni_t.shape[0]):
                v_idx = uni_t[k]
                n_triangles[triangles == v_idx] = k
            m_verts = vertices[uni_t]
            m_colors = vertex_colors[uni_t]
            nmesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(m_verts),
                triangles=o3d.utility.Vector3iVector(n_triangles)
                )
            nmesh.vertex_colors = o3d.utility.Vector3dVector(m_colors)

            meshes.append(nmesh)
        return par_v, meshes
    else:
        for i in range(len(c_list)):
            comp = c_list[i]
            n_sp_comp = len(comp)
            for j in range(n_sp_comp):
                P_idxs = comp[j][1]
                par_v[P_idxs] = i + 1    
        return par_v


def delete(P, idxs):
    if idxs.shape[0] == 0:
        return P
    P = np.delete(P, idxs, axis=0)
    return P


def rotate(P):
    axis = input("Axis [x, y, z]: ")
    if axis != "x" and axis != "y" and axis != "z":
        return P
    degree = input("Degree [-180, 180]: ")
    try:
        degree = float(degree)
    except:
        return P
    if degree < -180 or degree > 180:
        return P
    r = Rotation.from_euler(axis, degree, degrees=True)
    if type(P) == o3d.geometry.TriangleMesh:
        verts = np.asarray(P.vertices)
        verts = r.apply(verts)
        P.vertices = o3d.utility.Vector3dVector(verts)
    else:
        P[:, :3] = r.apply(P[:, :3])
    return P


def recenter(P):
    if type(P) == o3d.geometry.TriangleMesh:
        verts = np.asarray(P.vertices)
        center = np.mean(verts, axis=0)
        verts -= center
        P.vertices = o3d.utility.Vector3dVector(verts)
    else:
        center = np.mean(P[:, :3], axis=0)
        P[:, :3] -= center
    return P


def merge_small_singles(graph_dict, sp_idxs, unions, P, thres):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    n_edges = int(senders.shape[0])
    c_list = comp_list(graph_dict=graph_dict, unions=unions, n_P=P.shape[0], sp_idxs=sp_idxs)
    single_sps = []
    for i in range(len(c_list)):
        comp = c_list[i]
        n_sp_comp = len(comp)
        if n_sp_comp != 1:
            continue
        sp = comp[0]
        sp_idx = sp[0]
        single_sps.append(sp_idx)
    print("{0} single superpoints".format(len(single_sps)))
    n_unified = 0
    for i in range(n_edges):
        vi = int(senders[i])
        vj = int(receivers[i])
        sp_i = sp_idxs[vi]    
        sp_j = sp_idxs[vj]
        if sp_i.shape[0] < thres and sp_j.shape[0] < thres and vi in single_sps and vj in single_sps:
            unions[i] = True
            n_unified += 1
    print("Unified {0} edges".format(n_unified))
    return unions


def superpoint_info(picked_points_idxs, sp_idxs, P, graph_dict):
    if len(picked_points_idxs) != 1:
        return
    sp_info = uni_superpoint_idxs(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
    sp_info = sp_info[0]
    P_idxs = sp_idxs[sp_info]
    print("Superpoint: {0}".format(sp_info))
    print("Size: {0}".format(P_idxs.shape[0]))
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    n_edges = int(senders.shape[0])
    print("Total number of edges: {0}".format(n_edges))
    s_idxs = np.where(senders[:n_edges] == sp_info)[0]
    r_idxs = np.where(receivers[:n_edges] == sp_info)[0]
    print("Num senders: {0}, num receivers: {1}".format(s_idxs.shape[0], r_idxs.shape[0]))
    sp_senders = np.hstack((s_idxs[:, None], senders[s_idxs][:, None], receivers[s_idxs][:, None]))
    print("Sender edges (Idxs, Sender, Receivers):\n{0}".format(sp_senders))
    sp_receivers = np.hstack((r_idxs[:, None], senders[r_idxs][:, None], receivers[r_idxs][:, None]))
    print("Receiver edges (Idxs, Sender, Receivers):\n{0}".format(sp_receivers))

    return P[P_idxs]


def set_recursion(n_set_i, components, member, done):
    if member in done:
        return
    else:
        done.append(member)
    set_j = components[member]
    #print(member, set_j)
    n_set_i.update(set_j)
    for member_j in set_j:
        set_recursion(n_set_i=n_set_i, components=components, member=member_j, done=done)


def unions_to_partition(graph_dict, unions, sp_idxs, P, half=False):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    if half:
        n_edges = int(senders.shape[0] / 2)
    else:
        n_edges = int(senders.shape[0])
    components = len(sp_idxs) * [None]
    #print(len(sp_idxs))
    #return
    for i in range(len(sp_idxs)):
        components[i] = set()
        components[i].update([i])
    #
    for i in range(unions.shape[0]):
        union = unions[i]
        S_i = senders[i]
        S_j = receivers[i]
        if union:
            #print(S_i, S_j)
            components[S_i].update([int(S_j)])
            components[S_j].update([int(S_i)])
    #
    done = []
    c_components = []
    for i in range(len(components)):
        if i in done:
            #print("skip {0}".format(i))
            continue
        set_i = components[i]
        n_set_i = set_i.copy()
        set_done = [i]
        #print("---- SP: {0}, {1}".format(i, set_i))
        for member in set_i:
            set_recursion(n_set_i=n_set_i, components=components, member=member, done=set_done)
        #print(n_set_i)
        c_components.append(n_set_i)
        done.extend(list(n_set_i))
        #print(set_done)
    print("Found {0} connected components".format(len(c_components)))
    for i in range(0, len(c_components)):
        cc_i = c_components[i]
        for j in range(i, len(c_components)):
            if i == j:
                continue
            cc_j = c_components[j]
            cc_k = cc_i.intersection(cc_j)
            if len(cc_k) > 0:
                print(i, j, cc_k, len(cc_i), len(cc_j), len(cc_k))
                raise Exception("Error")
                #pass
    partition_vec = np.zeros((P.shape[0], ), dtype=np.uint32)
    p_n = 0
    for cc in c_components:
        #print(cc)
        for member in cc:
            P_idxs = sp_idxs[member]
            partition_vec[P_idxs] = p_n
        p_n += 1
    return partition_vec


def simplify_mesh(mesh):
    t = input("Target number of tris: ")
    try:
        t = int(t)
    except:
        return mesh
    n_tris_old = np.asarray(mesh.triangles).shape[0]
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=t)
    n_tris_new = np.asarray(mesh.triangles).shape[0]
    print("Simplified mesh from {0} to {1} triangles.".format(n_tris_old, n_tris_new))
    return mesh