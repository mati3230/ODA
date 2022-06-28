import argparse
import os
from tqdm import tqdm
import numpy as np

#"""
from scannet_utils import get_scenes, get_ground_truth
from ai_utils import graph, predict
from partition.felzenszwalb import partition_from_probs
from partition.partition import Partition
from partition.density_utils import densities_np
from exp_utils import \
    binary_cross_entropy,\
    predict_correl,\
    get_unions,\
    match_score,\
    get_csv_header,\
    save_csv,\
    mkdir,\
    load_exp_data,\
    match_score
#"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=100, help="k parameter of FH04 segmentation algorithm.")
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    #
    parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--csv_dir", default="./csvs_correl_vs_gnn", type=str, help="Directory where we save the csv.")
    parser.add_argument("--load_exp", default=False, type=bool, help="Use data from pre calculated dataset")
    parser.add_argument("--h5_dir", default="./exp_data", type=str, help="Directory where we load the h5 files.")
    args = parser.parse_args()

    mkdir(args.csv_dir)

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    n_scenes = len(scenes)
    """

            ooa_gnn, size_gnn, bce_gnn, ms_gnn, raw_score_gnn, 
            match_stats_gnn[0], match_stats_gnn[1], match_stats_gnn[2], 
            dens_stats_gnn[0], dens_stats_gnn[1], dens_stats_gnn[2],
    """
    csv_header = get_csv_header(header=["Name", "|P|"], algorithms=["GNN", "Correl"],
        algo_stats=["OOA", "|S|", "BCE", "MS", "RS", "NFOM", "NSOM", "NTOM", "DFOM", "DSOM", "DTOM"])
    csv_data = []
    desc = "Correlation vs. GNN"
    verbose = False
    model = None
    for i in tqdm(range(n_scenes), desc=desc, disable=verbose):
        scene_name = scenes[i]
        mesh, p_vec_gt, file_gt = get_ground_truth(scannet_dir=scannet_dir, scene=scene_name)
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
            probs_gnn = exp_dict["probs_gnn"]
            probs_correl = exp_dict["probs_correl"]
            bce_gnn = exp_dict["bce_gnn"][0]
            bce_correl = exp_dict["bce_correl"][0]
            p_vec_gt = exp_dict["p_gt"]
        else:
            p_gt = Partition(partition=p_vec_gt)

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

            p_cp = Partition(partition=part_cp)

            densities = p_gt.compute_densities(p_cp, densities_np)
            alpha = p_cp.alpha(densities)
            unions_gt = get_unions(graph_dict=graph_dict, alpha=alpha)

            if model is None:
                _, probs_gnn, model = predict(graph_dict=graph_dict, dec_b=0.8, return_model=True, verbose=False)
            else:
                _, probs_gnn = predict(graph_dict=graph_dict, dec_b=0.8, model=model, verbose=False)
            probs_correl = predict_correl(graph_dict=graph_dict)
            bce_gnn = binary_cross_entropy(y=unions_gt, probs=probs_gnn)
            bce_correl = binary_cross_entropy(y=unions_gt, probs=probs_correl)
        
        part_gnn = partition_from_probs(graph_dict=graph_dict, sim_probs=probs_gnn, k=args.k, P=P, sp_idxs=sp_idxs)
        part_correl = partition_from_probs(graph_dict=graph_dict, sim_probs=probs_correl, k=args.k, P=P, sp_idxs=sp_idxs)

        partition_gt = Partition(partition=p_vec_gt)
        ms_gnn, raw_score_gnn, ooa_gnn, match_stats_gnn, dens_stats_gnn = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_gnn), return_ooa=True)
        ms_correl, raw_score_correl, ooa_correl, match_stats_correl, dens_stats_correl = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_gnn), return_ooa=True)

        size_gnn = np.unique(part_gnn).shape[0]
        size_correl = np.unique(part_correl).shape[0]

        csv_data.append([file_gt, P.shape[0], 
            ooa_gnn, size_gnn, bce_gnn, ms_gnn, raw_score_gnn, 
            match_stats_gnn[0], match_stats_gnn[1], match_stats_gnn[2], 
            dens_stats_gnn[0], dens_stats_gnn[1], dens_stats_gnn[2],
            #
            ooa_correl, size_correl, bce_correl, ms_correl, raw_score_correl, 
            match_stats_correl[0], match_stats_correl[1], match_stats_correl[2], 
            dens_stats_correl[0], dens_stats_correl[1], dens_stats_correl[2]
            ])
        if i % args.pkg_size == 0 and len(csv_data) > 0:
            save_csv(res=csv_data, csv_dir=args.csv_dir, csv_name=str(i), data_header=csv_header)
            csv_data.clear()

        
if __name__ == "__main__":
    main()
    #test_bce()