import argparse
import itertools
from tqdm import tqdm
from multiprocessing import Pool
from p_tqdm import p_map
from random import shuffle, seed
import traceback

import open3d as o3d
import numpy as np

from io_utils import save_partitions,\
                    write_csv,\
                    mkdir
from ai_utils import superpoint_graph_mesh,\
                    superpoint_graph
from sp_utils import get_P,\
                    initial_partition
from scannet_utils import get_scenes, \
                    get_ground_truth
from partition.partition import Partition
from partition.density_utils import densities_np


def save_csv(res, csv_dir, csv_name, data_header):
    # write results of the cut pursuit calculations as csv
    csv = ""
    for header in data_header:
        csv += header + ","
    csv = csv[:-1] + "\n"

    for tup in res:
        if tup is None:
            continue
        for elem in tup:
            if type(elem) == list:
                csv += str(elem)[1:-1].replace(" ", "") + ","
            else:
                csv += str(elem) + ","
        csv = csv[:-1] + "\n"
    write_csv(filedir=csv_dir, filename=csv_name, csv=csv)


def ooa(par_v_gt, par_v_M, par_v_M_k, par_v_M_ks, par_v_P, nn_only):
    # calculate the OOA of multiple partitions
    sortation = np.argsort(par_v_gt)
    par_v_gt = par_v_gt[sortation]
    par_v_M = par_v_M[sortation]
    if not nn_only:
        par_v_M_k = par_v_M_k[sortation]
        par_v_M_ks = par_v_M_ks[sortation]
    
    par_v_P = par_v_P[sortation]

    ugt, ugt_idxs, ugt_counts = np.unique(par_v_gt, return_index=True, return_counts=True)

    partition_gt = Partition(partition=par_v_gt, uni=ugt, idxs=ugt_idxs, counts=ugt_counts)
    partition_M = Partition(partition=par_v_M)
    
    if not nn_only:
        partition_M_k = Partition(partition=par_v_M_k)
        partition_M_ks = Partition(partition=par_v_M_ks)
    
    partition_P = Partition(partition=par_v_P)

    # nr of vertices/points
    max_density = partition_gt.partition.shape[0]

    ooa_M, _ = partition_gt.overall_obj_acc(max_density=max_density, partition_B=partition_M, density_function=densities_np)
    
    if not nn_only:
        ooa_M_k, _ = partition_gt.overall_obj_acc(max_density=max_density, partition_B=partition_M_k, density_function=densities_np)
        ooa_M_ks, _ = partition_gt.overall_obj_acc(max_density=max_density, partition_B=partition_M_ks, density_function=densities_np)
    
    ooa_P, _ = partition_gt.overall_obj_acc(max_density=max_density, partition_B=partition_P, density_function=densities_np)

    if nn_only:
        return ooa_M, ooa_P
    else:
        return ooa_M, ooa_M_k, ooa_M_ks, ooa_P


def compare(comp_args):
    """ Method that executes the cut pursuit with different input graphs.
    
    Returns
    -------
    Statistics of the cut pursuit calculations. 

    """
    scene_id, scene_name, scannet_dir, reg_strength, k_nn_adj, partition_dir, nn_only, with_ooa, with_graph_stats, cross_only = comp_args
    #scene_name = "scene0085_01"
    lambda_edge_weight = 1.
    d_se_max = 0
    k_nn_adj = int(k_nn_adj)
    try:
        # Load the ground truth data (vertices, triangles, \Omega) 
        mesh, par_v_gt, file_gt = get_ground_truth(scannet_dir=scannet_dir, scene=scene_name)
        mesh.compute_adjacency_list()

        #print("")
        #print("1")
        #"""
        if cross_only:
            # calculate e_g
            P, xyz, rgb = get_P(mesh=mesh)
            n_sps_P0, n_sedg_P0, sp_idxs_P0, senders_P0, receivers_P0, _, stats_P0, graph_nn = superpoint_graph(
                xyz=P[:, :3],
                rgb=P[:, 3:],
                k_nn_adj=k_nn_adj,
                k_nn_geof=k_nn_adj,
                lambda_edge_weight=lambda_edge_weight,
                reg_strength=reg_strength,
                d_se_max=d_se_max,
                verbose=False,
                with_graph_stats=with_graph_stats,
                use_mesh_feat=True,
                use_mesh_graph=False,
                mesh_tris=mesh.triangles,
                adj_list=mesh.adjacency_list,
                g_dir="./tmp",
                g_filename=scene_name,
                n_proc=1,
                return_graph_nn=True,
                ignore_knn=True
                )
            # calculate g_e
            n_sps_P1, n_sedg_P1, sp_idxs_P1, senders_P1, receivers_P1, _, stats_P1 = superpoint_graph(
                xyz=P[:, :3],
                rgb=P[:, 3:],
                k_nn_adj=k_nn_adj,
                k_nn_geof=k_nn_adj,
                lambda_edge_weight=lambda_edge_weight,
                reg_strength=reg_strength,
                d_se_max=d_se_max,
                verbose=False,
                with_graph_stats=with_graph_stats,
                use_mesh_feat=False,
                use_mesh_graph=True,
                mesh_tris=mesh.triangles,
                adj_list=mesh.adjacency_list,
                g_dir="./tmp",
                g_filename=scene_name,
                n_proc=1,
                igraph_nn=graph_nn,
                ignore_knn=True
                )
            # return the result
            return scene_id, scene_name, P.shape[0], np.asarray(mesh.triangles).shape[0], reg_strength, k_nn_adj,\
                n_sps_P0, n_sedg_P0, n_sps_P1, n_sedg_P1, file_gt, stats_P0, stats_P1
    
        # calculate g
        n_sps_M, n_sedg_M, sp_idxs_M, senders_M, receivers_M, stris_M, stats_M = superpoint_graph_mesh(
            mesh_vertices_xyz=mesh.vertices,
            mesh_vertices_rgb=mesh.vertex_colors,
            mesh_tris=mesh.triangles,
            adj_list=mesh.adjacency_list,
            lambda_edge_weight=lambda_edge_weight,
            reg_strength=reg_strength,
            d_se_max=d_se_max,
            k_nn_adj=k_nn_adj,
            use_cartesian=False,
            n_proc=1,
            ignore_knn=False,
            verbose=False,
            g_dir="./tmp",
            g_filename=scene_name,
            with_graph_stats=with_graph_stats
            )
        #return None
        par_v_M = initial_partition(P=mesh, sp_idxs=sp_idxs_M, verbose=False)

        #"""
        if not nn_only:
            #print("2")
            # calculate M
            n_sps_M_k, n_sedg_M_k, sp_idxs_M_k, senders_M_k, receivers_M_k, stris_M_k, stats_M_k = superpoint_graph_mesh(
                mesh_vertices_xyz=mesh.vertices,
                mesh_vertices_rgb=mesh.vertex_colors,
                mesh_tris=mesh.triangles,
                adj_list=mesh.adjacency_list,
                lambda_edge_weight=lambda_edge_weight,
                reg_strength=reg_strength,
                d_se_max=d_se_max,
                k_nn_adj=k_nn_adj,
                use_cartesian=False,
                n_proc=1,
                ignore_knn=True,
                smooth=False,
                verbose=False,
                g_dir="./tmp",
                g_filename=scene_name,
                with_graph_stats=with_graph_stats
                )
            par_v_M_k = initial_partition(P=mesh, sp_idxs=sp_idxs_M_k, verbose=False)
            
            # calculate M with feature of g (experimental)
            n_sps_M_ks, n_sedg_M_ks, sp_idxs_M_ks, senders_M_ks, receivers_M_ks, stris_M_ks, stats_M_ks = superpoint_graph_mesh(
                mesh_vertices_xyz=mesh.vertices,
                mesh_vertices_rgb=mesh.vertex_colors,
                mesh_tris=mesh.triangles,
                adj_list=mesh.adjacency_list,
                lambda_edge_weight=lambda_edge_weight,
                reg_strength=reg_strength,
                d_se_max=d_se_max,
                k_nn_adj=k_nn_adj,
                use_cartesian=False,
                n_proc=1,
                ignore_knn=True,
                smooth=True,
                verbose=False,
                g_dir="./tmp",
                g_filename=scene_name,
                with_graph_stats=with_graph_stats
                )
            par_v_M_ks = initial_partition(P=mesh, sp_idxs=sp_idxs_M_ks, verbose=False)

        #print("4")
        # calculate e
        P, xyz, rgb = get_P(mesh=mesh)
        n_sps_P, n_sedg_P, sp_idxs_P, senders_P, receivers_P, _, stats_P = superpoint_graph(
            xyz=P[:, :3],
            rgb=P[:, 3:],
            k_nn_adj=k_nn_adj,
            k_nn_geof=k_nn_adj,
            lambda_edge_weight=lambda_edge_weight,
            reg_strength=reg_strength,
            d_se_max=d_se_max,
            verbose=False,
            with_graph_stats=with_graph_stats)
        par_v_P = initial_partition(P=P, sp_idxs=sp_idxs_P, verbose=False)

        # calculate the OOA (caution: slow!)
        if with_ooa:
            if nn_only:
                ooa_M, ooa_P = ooa(
                    par_v_gt=np.array(par_v_gt, copy=True),
                    par_v_M=np.array(par_v_M, copy=True),
                    par_v_M_k=None,
                    par_v_M_ks=None,
                    par_v_P=np.array(par_v_P, copy=True),
                    nn_only=nn_only)
            else:
                ooa_M, ooa_M_k, ooa_M_ks, ooa_P = ooa(
                    par_v_gt=np.array(par_v_gt, copy=True),
                    par_v_M=np.array(par_v_M, copy=True),
                    par_v_M_k=np.array(par_v_M_k, copy=True),
                    par_v_M_ks=np.array(par_v_M_ks, copy=True),
                    par_v_P=np.array(par_v_P, copy=True),
                    nn_only=nn_only)

        # how many outgoing edges do we have in M per vertex
        adj_list_ = mesh.adjacency_list.copy()
        n_per_v = np.zeros((len(adj_list_), ), np.uint32)
        for i in range(len(adj_list_)):
            al = adj_list_[i]
            n_per_v[i] = len(al)

        # save the partitions for later use
        partition_file = str(scene_id) + "_" + str(reg_strength) + "_" + str(k_nn_adj)
        if nn_only:
            partitions = [("gt", par_v_gt), ("M", par_v_M), ("P", par_v_P)]
        else:
            partitions = [("gt", par_v_gt), ("M", par_v_M), ("P", par_v_P), ("M_k", par_v_M_k), ("M_ks", par_v_M_ks)]
        save_partitions(partitions=partitions, fdir=partition_dir, fname=partition_file, verbose=False)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None

    # return the stats depending on the configuration
    if with_ooa:
        if nn_only:
            return scene_id, scene_name, P.shape[0], np.asarray(mesh.triangles).shape[0], reg_strength, k_nn_adj,\
                n_sps_M, n_sedg_M, ooa_M, n_sps_P, n_sedg_P, ooa_P,\
                np.mean(n_per_v), np.std(n_per_v), np.median(n_per_v), file_gt, partition_file, stats_M, stats_P
        else:
            return scene_id, scene_name, P.shape[0], np.asarray(mesh.triangles).shape[0], reg_strength, k_nn_adj,\
                n_sps_M, n_sedg_M, ooa_M, n_sps_M_k, n_sedg_M_k, ooa_M_k, n_sps_M_ks, n_sedg_M_ks, ooa_M_ks, n_sps_P, n_sedg_P, ooa_P,\
                np.mean(n_per_v), np.std(n_per_v), np.median(n_per_v), file_gt, partition_file, stats_M, stats_M_k, stats_M_ks, stats_P
    else:
        if nn_only:
            return scene_id, scene_name, P.shape[0], np.asarray(mesh.triangles).shape[0], reg_strength, k_nn_adj,\
                n_sps_M, n_sedg_M, n_sps_P, n_sedg_P,\
                np.mean(n_per_v), np.std(n_per_v), np.median(n_per_v), file_gt, partition_file, stats_M, stats_P
        else:
            return scene_id, scene_name, P.shape[0], np.asarray(mesh.triangles).shape[0], reg_strength, k_nn_adj,\
                n_sps_M, n_sedg_M, n_sps_M_k, n_sedg_M_k, n_sps_M_ks, n_sedg_M_ks, n_sps_P, n_sedg_P,\
                np.mean(n_per_v), np.std(n_per_v), np.median(n_per_v), file_gt, partition_file, stats_M, stats_M_k, stats_M_ks, stats_P
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_dir", default="./partitions_files", type=str, help="Directory where we save the partiton files.")
    parser.add_argument("--csv_dir", default="./csvs", type=str, help="Directory where we save the csv.")
    parser.add_argument("--csv_name", default="mesh_cloud_results", type=str, help="filename of the csv.")
    parser.add_argument("--n_proc", default=1, type=int, help="Number of processes that will be used.")
    parser.add_argument("--pkg_size", default=100, type=int, help="Number of packages to save a csv")
    parser.add_argument("--offset", default=0, type=int, help="Offset for the name of the csv file.")
    parser.add_argument("--nn_only", default=False, type=bool, help="If True, only geodesic and euclidean nearest neighbours will be calculated.")
    parser.add_argument("--m_ids", default=False, type=bool, help="Use user defined scene ids (see Line 455 in this script).")
    parser.add_argument("--without_ooa", default=False, type=bool, help="Disable the OOA calculation")
    parser.add_argument("--with_graph_stats", default=False, type=bool, help="Enable the computation of mean features, weights, ...")
    parser.add_argument("--cross_only", default=False, type=bool, help="Calculate graph e with features of g (e_g) and vice versa (g_e)")
    args = parser.parse_args()
    print(args)
    mkdir(args.partition_dir)
    mkdir(args.csv_dir)
    seed(42)

    with_ooa = not args.without_ooa
    cross_only = args.cross_only
    nn_only = args.nn_only
    with_graph_stats = args.with_graph_stats

    # setup of the header
    if cross_only:
        with_ooa = False
        nn_only = False
        with_graph_stats = True

    data_header = [
        "id", # integer that is mapped to the scene
        "name", # name of the scene
        "|V|", # nr of vertices
        "|T|", # nr of triangles
        "lambda", # regularization strength of cp
        "knn" # neighbourhood size which is considered in the CP
    ]
    if cross_only:
        data_header.extend([
            "|S_M|", # nr of superpoints in the mesh partition
            "|E_M|" # nr of superedges in the mesh superpoint graph
        ])
    else:
        data_header.extend([
            "|P0_M|", # nr of superpoints in the mesh partition
            "|P0_M|" # nr of superedges in the mesh superpoint graph
        ])

    if with_ooa:
        data_header.append("OOA_M") # overall object accuracy of the mesh

    if not nn_only:
        data_header.extend([
            "|S_M_k|", # nr of superpoints in the mesh partition
            "|E_M_k|" # nr of superedges in the mesh superpoint graph
        ])
        if with_ooa:
            data_header.append("OOA_M_k") # overall object accuracy of the mesh
        data_header.extend([
            "|S_M_ks|", # nr of superpoints in the mesh partition
            "|E_M_ks|" # nr of superedges in the mesh superpoint graph
        ])
        if with_ooa:
            data_header.append("OOA_M_ks") # overall object accuracy of the mesh
    if cross_only:
        data_header.extend([
            "|S_P1|", # nr of superpoints in the point cloud partition
            "|E_P1|" # nr of superedges in the point cloud superpoint graph
        ])
    else:
        data_header.extend([
            "|S_P|", # nr of superpoints in the point cloud partition
            "|E_P|" # nr of superedges in the point cloud superpoint graph
        ])
    if with_ooa:
        data_header.append("OOA_P") # overall object accuracy of the point cloud
    if not cross_only:
        data_header.extend([
            "mean(n)", # average number of neighbours per vertex
            "std(n)", # std of neighbours per vertex
            "median(n)"# median of neighbours per vertex
        ])
    data_header.append("file_gt")# path to the ground truth mesh
    if not cross_only:
        data_header.append("partitions_file") # path to a file where the partitions are stored

    if nn_only:
        posts = ["_M", "_P"]
    else:
        posts = ["_M", "_M_k", "_M_ks", "_P"]

    if cross_only:
        posts = ["_P0", "_P1"]

    for post in posts:
        data_header.append("max_ite" + post)
        data_header.append("n_ite" + post)
        data_header.append("exit_code" + post)

        data_header.append("s1" + post)
        data_header.append("s2" + post)
        data_header.append("s3" + post)
        data_header.append("s4" + post)
        data_header.append("s5" + post)

        data_header.append("e1_1" + post)
        data_header.append("e1_2" + post)
        data_header.append("e1_3" + post)
        data_header.append("e1_4" + post)
        data_header.append("e1_5" + post)

        data_header.append("e2_1" + post)
        data_header.append("e2_2" + post)
        data_header.append("e2_3" + post)
        data_header.append("e2_4" + post)
        data_header.append("e2_5" + post)

        data_header.append("|SSp1|" + post)
        data_header.append("|SSp2|" + post)
        data_header.append("|SSp3|" + post)
        data_header.append("|SSp4|" + post)
        data_header.append("|SSp5|" + post)

        data_header.append("|SR1|" + post)
        data_header.append("|SR2|" + post)
        data_header.append("|SR3|" + post)
        data_header.append("|SR4|" + post)
        data_header.append("|SR5|" + post)
        if with_graph_stats:
            for feat in ["L", "P", "S", "V"]:
                data_header.append("mean({0}){1}".format(feat, post))
            for feat in ["L", "P", "S", "V"]:
                data_header.append("std({0}){1}".format(feat, post))
            for feat in ["L", "P", "S", "V"]:
                data_header.append("median({0}){1}".format(feat, post))
            data_header.append("mean(w){0}".format(post))
            data_header.append("std(w){0}".format(post))
            data_header.append("median(w){0}".format(post))

    # lists with parameter configurations
    knns = [30]
    reg_strengths = [0.1]

    # create tuple with parameter configurations
    cp_args = list(itertools.product(*[reg_strengths, knns]))
    n_cp_args = len(cp_args)

    # get the list of scannet scenes
    scenes, scannet_dir = get_scenes()
    #scenes = ["s1"]
    #scannet_dir = ""
    n_scenes = len(scenes)
    #n_scenes=2
    #n_cp_args=2

    if args.m_ids: # if using user defined scenes in contrast to all scenes that are randomized
        # user defined scenes
        m_scene_ids = [1456, 510, 618, 1453, 1455, 1507, 325, 1240, 1260, 515,
            1323, 687, 1282, 1390, 1439, 550, 1296, 593, 373, 1418, 353, 549,
            731, 1329, 400, 759, 1065, 1265, 782, 1057, 1352, 718, 886, 1387,
            483, 1249, 428, 547, 964, 463, 1297, 1412, 1095, 182, 293, 385,
            1151, 969, 111, 926, 1322, 1174, 1434, 386, 1053, 482, 1305, 215, 857,
            794, 961, 207, 132, 1015, 1241, 288, 887, 68, 297, 580, 1041, 787,
            932, 1420, 1435, 413, 1359, 1345, 1298, 626, 715]
        # list where each element is a job that should be computed
        comp_args = (n_cp_args*len(m_scene_ids)) * [None]

        scenes_ids = list(zip(scenes, list(range(len(scenes)))))

        shuffle(scenes_ids)

        # fill the list of jobs
        idx = 0
        for (scene_name, scene_id) in scenes_ids:
            if scene_id not in m_scene_ids:
                continue
            for cp_id in range(n_cp_args):
                #idx = n_cp_args * scene_id + cp_id
                reg_strength, k_nn_adj = cp_args[cp_id]
                comp_args[idx] = (scene_id, scene_name, scannet_dir, reg_strength,\
                    k_nn_adj, args.partition_dir, nn_only, with_ooa, with_graph_stats, cross_only)
                idx += 1
    else:
        comp_args = (n_cp_args*n_scenes) * [None]

        #partition_file = str(scene_id) + "_" + str(reg_strength) + "_" + str(k_nn_adj)

        scenes_ids = list(zip(scenes, list(range(len(scenes)))))

        shuffle(scenes_ids)

        idx = 0
        for (scene_name, scene_id) in scenes_ids:
            for cp_id in range(n_cp_args):
                #idx = n_cp_args * scene_id + cp_id
                reg_strength, k_nn_adj = cp_args[cp_id]
                comp_args[idx] = (scene_id, scene_name, scannet_dir, reg_strength, k_nn_adj, args.partition_dir, nn_only, with_ooa, with_graph_stats, cross_only)
                idx += 1
    print("")
    print(data_header)
    print("")
    print(scenes_ids[:10])
    if args.n_proc == 1:
        res = []
        for i in tqdm(range(len(comp_args)), desc="Compare"):
            comp_arg = comp_args[i] 
            # this method computes a configuration
            r = compare(comp_args=comp_arg)
            #print("result:", r)
            if r is None:
                continue
            res.append(r)
            if i % args.pkg_size == 0:
                save_csv(res=res, csv_dir=args.csv_dir, csv_name=args.csv_name + "_" + str(i+args.offset), data_header=data_header)
                res = []
    elif args.n_proc > 1: # compute multiple configuration at once
        """print("Use {0} processes".format(args.n_proc))
        with Pool(processes=args.n_proc) as p:
            res = p.starmap(compare, comp_args)"""
        idxs = [i for i in range(0, len(comp_args), args.pkg_size)]
        for i in idxs:
            res = p_map(compare, comp_args[i:i+args.pkg_size], num_cpus=args.n_proc)
            save_csv(res=res, csv_dir=args.csv_dir, csv_name=args.csv_name + "_" + str(i+args.offset), data_header=data_header)


if __name__ == "__main__":
    main()