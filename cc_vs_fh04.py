import argparse
import os
from tqdm import tqdm
import numpy as np

from scannet_utils import get_scenes, get_ground_truth
from ai_utils import graph, predict
from partition.felzenszwalb import partition_from_probs
from sp_utils import partition
from partition.partition import Partition
from partition.density_utils import densities_np
from exp_utils import \
    match_score,\
    get_csv_header,\
    save_csv,\
    mkdir,\
    load_exp_data

from visu_utils import render_partition_vec_o3d
from io_utils import load_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=0.8, help="Similarity threshold.")
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
    parser.add_argument("--csv_dir", default="./csvs_cc_vs_fh04", type=str, help="Directory where we save the csv.")
    parser.add_argument("--load_exp", default=False, type=bool, help="Use data from pre calculated dataset")
    parser.add_argument("--h5_dir", default="./exp_data", type=str, help="Directory where we load the h5 files.")
    args = parser.parse_args()

    colors = load_colors()
    colors = colors/255.

    mkdir(args.csv_dir)

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    n_scenes = len(scenes)
    
    csv_header = get_csv_header(header=["Name", "|P|"], algorithms=["CP", "FH04", "FH04_GT", "CC", "CC_GT"],
        algo_stats=["OOA", "|S|", "MS", "RS", "NFOM", "NSOM", "NTOM", "DFOM", "DSOM", "DTOM"])
    csv_data = []
    desc = "CC vs. FH04"
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
            probs = exp_dict["probs_gnn"]
            p_vec_gt = exp_dict["p_gt"]
            part_cp = exp_dict["p_cp"]
            unions_gt = exp_dict["unions_gt"]
            probs_gt = unions_gt.astype(np.float32)
            unions = np.zeros(probs.shape, dtype=np.bool)
            for j in range(probs.shape[0]):
                prob = probs[j]
                unions[j] = prob >= args.t
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
                unions, probs, model = predict(graph_dict=graph_dict, dec_b=args.t, return_model=True, verbose=False)
            else:
                unions, probs = predict(graph_dict=graph_dict, dec_b=args.t, model=model, verbose=False)

        sortation = np.argsort(p_vec_gt)
        p_vec_gt = p_vec_gt[sortation]

        part_fh04 = partition_from_probs(graph_dict=graph_dict, sim_probs=probs, k=args.k, P=P, sp_idxs=sp_idxs)
        part_cc = partition(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, half=False, verbose=verbose)
        part_fh04_gt = partition_from_probs(graph_dict=graph_dict, sim_probs=probs_gt, k=args.k, P=P, sp_idxs=sp_idxs)
        part_cc_gt = partition(graph_dict=graph_dict, unions=unions_gt, P=P, sp_idxs=sp_idxs, half=False, verbose=verbose)
        #render_partition_vec_o3d(mesh=mesh, partition=part_cc_gt, colors=colors)

        part_fh04 = part_fh04[sortation]
        part_cc = part_cc[sortation]
        part_cp = part_cp[sortation]
        part_cc_gt = part_cc_gt[sortation]
        part_fh04_gt = part_fh04_gt[sortation]

        partition_gt = Partition(partition=p_vec_gt)
        ms_fh04, raw_score_fh04, ooa_fh04, match_stats_fh04, dens_stats_fh04 = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_fh04), return_ooa=True)
        ms_cc, raw_score_cc, ooa_cc, match_stats_cc, dens_stats_cc = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_cc), return_ooa=True)
        ms_cp, raw_score_cp, ooa_cp, match_stats_cp, dens_stats_cp = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_cp), return_ooa=True)
        ms_fh04_gt, raw_score_fh04_gt, ooa_fh04_gt, match_stats_fh04_gt, dens_stats_fh04_gt = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_fh04_gt), return_ooa=True)
        ms_cc_gt, raw_score_cc_gt, ooa_cc_gt, match_stats_cc_gt, dens_stats_cc_gt = match_score(
            gt_partition=partition_gt, partition=Partition(partition=part_cc_gt), return_ooa=True)

        size_fh04 = np.unique(part_fh04).shape[0]
        size_cc = np.unique(part_cc).shape[0]
        size_fh04_gt = np.unique(part_fh04_gt).shape[0]
        size_cc_gt = np.unique(part_cc_gt).shape[0]
        size_cp = len(sp_idxs)

        csv_data.append([file_gt, P.shape[0],
                ooa_cp, size_cp, ms_cp, raw_score_cp, 
                match_stats_cp[0], match_stats_cp[1], match_stats_cp[2], 
                dens_stats_cp[0], dens_stats_cp[1], dens_stats_cp[2],
                #
                ooa_fh04, size_fh04, ms_fh04, raw_score_fh04, 
                match_stats_fh04[0], match_stats_fh04[1], match_stats_fh04[2], 
                dens_stats_fh04[0], dens_stats_fh04[1], dens_stats_fh04[2],
                #
                ooa_fh04_gt, size_fh04_gt, ms_fh04_gt, raw_score_fh04_gt, 
                match_stats_fh04_gt[0], match_stats_fh04_gt[1], match_stats_fh04_gt[2], 
                dens_stats_fh04_gt[0], dens_stats_fh04_gt[1], dens_stats_fh04_gt[2],
                #
                ooa_cc, size_cc, ms_cc, raw_score_cc, 
                match_stats_cc[0], match_stats_cc[1], match_stats_cc[2], 
                dens_stats_cc[0], dens_stats_cc[1], dens_stats_cc[2],
                #
                ooa_cc_gt, size_cc_gt, ms_cc_gt, raw_score_cc_gt, 
                match_stats_cc_gt[0], match_stats_cc_gt[1], match_stats_cc_gt[2], 
                dens_stats_cc_gt[0], dens_stats_cc_gt[1], dens_stats_cc_gt[2]
            ])
        if i % args.pkg_size == 0 and len(csv_data) > 0:
            save_csv(res=csv_data, csv_dir=args.csv_dir, csv_name=str(i), data_header=csv_header)
            csv_data.clear()

        
if __name__ == "__main__":
    main()