import numpy as np
import igraph as ig
import open3d as o3d


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
            print("Add new edge from node {0} to {1}".format(sp_i, sp_j))
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


def to_igraph(graph_dict, unions):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    n_edges = senders.shape[0]
    edge_list = n_edges*[None]
    for i in range(n_edges):
        vi = int(senders[i])
        vj = int(receivers[i])
        edge_list[i] = (vi, vj)
    g = ig.Graph(edges=edge_list)
    g.es["union"] = unions.tolist()
    return g


def comp_list(graph_dict, unions, n_P, sp_idxs):
    ig_graph = to_igraph(graph_dict=graph_dict, unions=unions)
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
    components_list = comp_list(graph_dict=graph_dict, unions=unions, n_P=n_P, sp_idxs=sp_idxs)
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


def reconstruct_obj(P, depth):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(P[:, 3:] / 255.)

    pcd.estimate_normals()
    print("Normals estimated.")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    #radii = 3 * [0.1]
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)

    return mesh


def reconstruct(P, objects, remaining, depth):
    meshes = (len(objects) + 1) * [None]
    for i in range(len(objects)):
        o_idxs = objects[i]
        P_obj = P[o_idxs]
        mesh_obj = reconstruct_obj(P=P_obj, depth=depth)
        meshes[i] = mesh_obj
    P_r = P[remaining]
    mesh_r = reconstruct_obj(P=P_r, depth=depth)
    meshes[-1] = mesh_r
    return meshes


def initial_partition(P, sp_idxs):
    n_P = P.shape[0]
    par_v = np.zeros((n_P, ), dtype=np.uint32)
    n_sps = len(sp_idxs)
    print("Number of superpoints: {0}".format(n_sps))
    for i in range(n_sps):
        idxs = sp_idxs[i]
        par_v[idxs] = i+1
    return par_v


def partition(graph_dict, unions, P, sp_idxs):
    n_P = P.shape[0]
    par_v = np.zeros((n_P, ), dtype=np.uint32)
    c_list = comp_list(graph_dict=graph_dict, unions=unions, n_P=n_P, sp_idxs=sp_idxs)

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