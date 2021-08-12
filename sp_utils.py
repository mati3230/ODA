import numpy as np
import igraph as ig
import open3d as o3d


def unify(picked_points_idxs, sp_idxs, graph_dict, unions):
    if len(picked_points_idxs) == 0:
        return graph_dict, unions
    sp_to_unify = superpoint_idxs(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]
    n_sp_to_unify = len(sp_to_unify)

    idxs_to_del = np.zeros((0, 1), dtype=np.uint32)
    for i in range(n_sp_to_unify):
        sp_i = sp_to_unify[i]
        idxs = np.where(senders == sp_i)[0]
        idxs_to_del = np.vstack((idxs_to_del, idxs[:, None]))
        idxs = np.where(receivers == sp_i)[0]
        idxs_to_del = np.vstack((idxs_to_del, idxs[:, None]))
    idxs_to_del = idxs_to_del.reshape(idxs_to_del.shape[0], )
    idxs_to_del = np.unique(idxs_to_del)
    unions[idxs_to_del] = False
    
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
            n_edges += 1
    if n_edges > 0:
        n_senders = np.array(n_senders, dtype=np.uint32)
        senders = np.vstack((senders[:, None], n_senders[:, None]))
        senders = senders.reshape(senders.shape[0], )
        n_receivers = np.array(n_receivers, dtype=np.uint32)
        receivers = np.vstack((receivers[:, None], n_receivers[:, None]))
        receivers = senders.reshape(receivers.shape[0], )
        n_unions = np.ones((n_edges, 1), dtype=np.bool)
        unions = np.vstack((unions[:, None], n_unions))
        unions = unions.reshape(unions.shape[0], )
    graph_dict["senders"] = senders
    graph_dict["receivers"] = receivers

    return graph_dict, unions


def superpoint_idxs(picked_points_idxs, sp_idxs):
    result = []
    for i in range(len(picked_points_idxs)):
        p_idx = picked_points_idxs[i]
        for j in range(len(sp_idxs)):
            idxs = sp_idxs[j]
            if p_idx in idxs:
                result.append(j)
                break
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
    selected_sps = superpoint_idxs(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs)
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


def reconstruct_obj(P, alpha):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(P[:, 3:] / 255.)
    pcd.estimate_normals()

    #mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    #radii = [0.005, 0.01, 0.02, 0.04]
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)

    return mesh


def reconstruct(P, objects, remaining, alpha):
    meshes = (len(objects) + 1) * [None]
    for i in range(len(objects)):
        o_idxs = objects[i]
        P_obj = P[o_idxs]
        mesh_obj = reconstruct_obj(P=P_obj, alpha=alpha)
        meshes[i] = mesh_obj
    P_r = P[remaining]
    mesh_r = reconstruct_obj(P=P_r, alpha=alpha)
    meshes[-1] = mesh_r
    return meshes