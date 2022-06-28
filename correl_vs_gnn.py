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
    ooa,\
    get_csv_header,\
    save_csv,\
    mkdir
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
    args = parser.parse_args()

    mkdir(args.csv_dir)

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    n_scenes = len(scenes)
    
    csv_header = get_csv_header(header=["Name", "|P|"], algorithms=["GNN", "Correl"], algo_stats=["OOA", "|S|", "BCE"])
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
        part_gnn = partition_from_probs(graph_dict=graph_dict, sim_probs=probs_gnn, k=args.k, P=P, sp_idxs=sp_idxs)
        
        probs_correl = predict_correl(graph_dict=graph_dict)
        part_correl = partition_from_probs(graph_dict=graph_dict, sim_probs=probs_correl, k=args.k, P=P, sp_idxs=sp_idxs)

        ooa_gnn, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_gnn)
        ooa_correl, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_correl, partition_gt=partition_gt, sortation=sortation)

        size_gnn = np.unique(part_gnn).shape[0]
        size_correl = np.unique(part_correl).shape[0]

        bce_gnn = binary_cross_entropy(y=unions_gt, probs=probs_gnn)
        bce_correl = binary_cross_entropy(y=unions_gt, probs=probs_correl)

        csv_data.append([file_gt, P.shape[0], ooa_gnn, size_gnn, bce_gnn, ooa_correl, size_correl, bce_correl])
        if i % args.pkg_size == 0 and len(csv_data) > 0:
            save_csv(res=csv_data, csv_dir=args.csv_dir, csv_name=str(i), data_header=csv_header)
            csv_data.clear()

        
if __name__ == "__main__":
    main()
    #test_bce()