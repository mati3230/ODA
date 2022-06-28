import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scannet_utils import get_scenes, get_ground_truth
from ai_utils import graph, predict
from partition.felzenszwalb import partition_from_probs
from partition.partition import Partition
from exp_utils import \
    match_score,\
    get_csv_header,\
    save_csv,\
    mkdir,\
    load_exp_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_k", type=float, default=1000, help="Minimum k parameter of FH04 segmentation algorithm.")
    parser.add_argument("--max_k", type=float, default=100, help="Maximum k parameter of FH04 segmentation algorithm.")
    parser.add_argument("--step_k", type=float, default=100, help="Step size to which the k parameter of FH04 segmentation algorithm will be varied.")
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    #
    #parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--csv_dir", default="./", type=str, help="Directory where we save the csv.")
    #
    parser.add_argument("--sn_id", default=0, type=int, help="Scannet scene id in the range of [0,707]")
    args = parser.parse_args()

    if args.min_k >= args.max_k:
        raise Exception("min_k {0} should be smaller than max_k {1}".format(args.min_k, args.max_k))
    if args.step_k >= args.max_k:
        raise Exception("step_k {0} should be smaller than max_k {1}".format(args.step_k, args.max_k))
    k_params = np.arange(start=args.min_k, stop=args.max_k, step=args.step_k)

    max_sn_id = 707
    min_sn_id = 0
    if args.sn_id > max_sn_id or args.sn_id < min_sn_id:
        raise Exception("Scannet ID {0} should be between {1} and {2}.".format(args.sn_id, min_sn_id, max_sn_id))
    scenes, _, scannet_dir = get_scenes(blacklist=[])
    scene_name = scenes[args.sn_id]

    csv_header = get_csv_header()
    csv_name = "fh04_k_id{0}".format(args.sn_id)
    
    verbose = False
    model = None

    mesh, p_vec_gt, file_gt = get_ground_truth(scannet_dir=scannet_dir, scene=scene_name)
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    P = np.hstack((xyz, rgb))

    graph_dict, sp_idxs, part_cp = graph(
        cloud=P,
        k_nn_adj=args.k_nn_adj,
        k_nn_geof=args.k_nn_geof,
        lambda_edge_weight=args.lambda_edge_weight,
        reg_strength=args.reg_strength,
        d_se_max=args.d_se_max,
        max_sp_size=args.max_sp_size,
        verbose=verbose,
        return_p_vec=True)

    unions, probs, model = predict(graph_dict=graph_dict, dec_b=0.8, return_model=True, verbose=False)

    ooas = []
    sizes = []
    all_data = []
    partition_gt = None
    sortation = None

    ooa_cp, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_cp, partition_gt=partition_gt, sortation=sortation)
    size_cp = len(sp_idxs)

    for i in tqdm(range(k_params.shape[0]), desc="Vary K of FH04", disable=verbose):
        k = k_params[i]
        part_fh04 = partition_from_probs(graph_dict=graph_dict, sim_probs=probs, k=k, P=P, sp_idxs=sp_idxs)
        ooa_fh04, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_fh04, partition_gt=partition_gt, sortation=sortation)
        size_fh04 = np.unique(part_fh04).shape[0]
        ooas.append(ooa_fh04)
        sizes.append(size_fh04)
        all_data.append([scene_name, k, ooa_fh04, size_fh04, ooa_cp, size_cp])
    
    save_csv(res=all_data, csv_dir=args.csv_dir, csv_name=csv_name, data_header=csv_header)

    ooas = np.array(ooas)
    sizes = np.array(sizes)

    ooa_cps = np.zeros((ooas.shape[0], ))
    ooa_cps[:] = ooa_cp

    plt.plot(k_params, ooas, ".")
    plt.plot(k_params, ooa_cps, "-")
    #plt.xticks(k_params)
    plt.savefig(fname="fh04_k_ooa_id{0}.jpg".format(args.sn_id))
    plt.show()


    sizes_cps = np.zeros((sizes.shape[0], ))
    sizes_cps[:] = size_cp
    plt.plot(k_params, sizes, ".")
    plt.plot(k_params, sizes_cps, "-")
    #plt.xticks(k_params)
    plt.savefig(fname="fh04_k_sizes_id{0}.jpg".format(args.sn_id))
    plt.show()


def all_scenes():
    parser = argparse.ArgumentParser()
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    #
    parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--csv_dir", default="./csvs_vary_k", type=str, help="Directory where we save the csv.")
    parser.add_argument("--load_exp", default=False, type=bool, help="Use data from pre calculated dataset")
    parser.add_argument("--h5_dir", default="./exp_data", type=str, help="Directory where we load the h5 files.")
    args = parser.parse_args()

    mkdir(args.csv_dir)

    k_params = [0.001, 0.1, 1, 2, 10, 100]

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    
    csv_header = get_csv_header(header=["Name", "K"], algorithms=["FH04", "CP"],
        algo_stats=["OOA", "|S|", "MS", "RS", "NFOM", "NSOM", "NTOM", "DFOM", "DSOM", "DTOM"])
    
    verbose = False
    model = None

    for j in tqdm(range(len(scenes)), desc="Vary K"):
        scene_name = scenes[j]
        mesh, p_vec_gt, file_gt = get_ground_truth(
            scannet_dir=scannet_dir, scene=scene_name)
        xyz = np.asarray(mesh.vertices)
        rgb = np.asarray(mesh.vertex_colors)
        P = np.hstack((xyz, rgb))

        if args.load_exp:
            exp_dict, sp_idxs = load_exp_data(fdir=args.h5_dir, fname=scene_name)
            graph_dict = {
                "nodes": exp_dict["node_features"],
                "senders": exp_dict["senders"],
                "receivers": exp_dict["receivers"]
            }
            probs = exp_dict["probs_gnn"]
            p_vec_gt = exp_dict["p_gt"]
            part_cp = exp_dict["p_cp"]
        else:
            graph_dict, sp_idxs, part_cp = graph(
                cloud=P,
                k_nn_adj=args.k_nn_adj,
                k_nn_geof=args.k_nn_geof,
                lambda_edge_weight=args.lambda_edge_weight,
                reg_strength=args.reg_strength,
                d_se_max=args.d_se_max,
                max_sp_size=args.max_sp_size,
                verbose=verbose,
                return_p_vec=True)
            if model is None:
                _, probs, model = predict(graph_dict=graph_dict, dec_b=0.8, return_model=True, verbose=False)
            else:
                _, probs = predict(graph_dict=graph_dict, dec_b=0.8, model=model, verbose=False)

        ooas = []
        sizes = []
        all_data = []
        partition_gt = None

        sortation = np.argsort(p_vec_gt)
        p_vec_gt = p_vec_gt[sortation]
        part_cp = part_cp[sortation]
        partition_gt = Partition(partition=p_vec_gt)
        ms_cp, raw_score_cp, ooa_cp, match_stats_cp, dens_stats_cp = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_cp), return_ooa=True)
        size_cp = np.unique(part_cp).shape[0]

        for i in range(len(k_params)):
            k = k_params[i]
            part_fh04 = partition_from_probs(graph_dict=graph_dict, sim_probs=probs, k=k, P=P, sp_idxs=sp_idxs)
            part_fh04 = part_fh04[sortation]
            ms_fh04, raw_score_fh04, ooa_fh04, match_stats_fh04, dens_stats_fh04 = match_score(
                gt_partition=partition_gt, partition=Partition(partition=part_fh04), return_ooa=True)
            
            size_fh04 = np.unique(part_fh04).shape[0]
            ooas.append(ooa_fh04)
            sizes.append(size_fh04)
            all_data.append([
                scene_name, k, ooa_fh04, size_fh04, ms_fh04, raw_score_fh04, 
                match_stats_fh04[0], match_stats_fh04[1], match_stats_fh04[2], 
                dens_stats_fh04[0], dens_stats_fh04[1], dens_stats_fh04[2],
                #
                ooa_cp, size_cp, ms_cp, raw_score_cp, 
                match_stats_cp[0], match_stats_cp[1], match_stats_cp[2], 
                dens_stats_cp[0], dens_stats_cp[1], dens_stats_cp[2]
                ])
        save_csv(res=all_data, csv_dir=args.csv_dir, csv_name=str(j), data_header=csv_header)
        all_data.clear()


if __name__ == "__main__":
    #main()
    all_scenes()