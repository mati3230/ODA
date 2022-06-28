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
    ooa,\
    get_csv_header,\
    save_csv,\
    mkdir


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

    mkdir(args.csv_dir)

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    n_scenes = len(scenes)
    
    csv_header = get_csv_header(header=["Name", "|P|"], algorithms=["FH04", "CC", "CP"], algo_stats=["OOA", "|S|"])
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

        part_fh04 = partition_from_probs(graph_dict=graph_dict, sim_probs=probs, k=args.k, P=P, sp_idxs=sp_idxs)
        part_cc = partition(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, half=False, verbose=verbose)

        ooa_fh04, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_fh04)
        ooa_cc, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_cc, partition_gt=partition_gt, sortation=sortation)
        ooa_cp, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_cp, partition_gt=partition_gt, sortation=sortation)

        size_fh04 = np.unique(part_fh04).shape[0]
        size_cc = np.unique(part_cc).shape[0]
        size_cp = len(sp_idxs)

        csv_data.append([file_gt, P.shape[0], ooa_fh04, size_fh04, ooa_cc, size_cc, ooa_cp, size_cp])
        if i % args.pkg_size == 0 and len(csv_data) > 0:
            save_csv(res=csv_data, csv_dir=args.csv_dir, csv_name=str(i), data_header=csv_header)
            csv_data.clear()

        
if __name__ == "__main__":
    main()