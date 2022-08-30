import argparse
import os
from tqdm import tqdm
import numpy as np

from scannet_utils import get_scenes, get_ground_truth
from ai_utils import graph, predict
from exp_utils import mkdir, predict_correl, get_unions, get_imperfect_probs, save_exp_data, binary_cross_entropy, classification_metrics
from partition.partition import Partition
from partition.density_utils import densities_np
from visu_utils import render_graph, render_partition_vec_o3d
from sp_utils import sp_centers, partition
from io_utils import load_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=0.8, help="Similarity threshold.")
    #parser.add_argument("--k", type=float, default=100, help="k parameter of FH04 segmentation algorithm.")
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    #
    #parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--h5_dir", default="./exp_data", type=str, help="Directory where we save the h5 files.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    mkdir(args.h5_dir)

    colors = load_colors()
    colors = colors/255.

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    n_scenes = len(scenes)
    
    desc = "Exp Data"
    verbose = False
    model = None
    for i in tqdm(range(n_scenes), desc=desc, disable=verbose):
        scene_name = scenes[i]
        mesh, p_vec_gt, file_gt = get_ground_truth(scannet_dir=scannet_dir, scene=scene_name)
        sortation = np.argsort(p_vec_gt)
        sortation = sortation.astype(np.uint32)
        xyz = np.asarray(mesh.vertices)[sortation]
        rgb = np.asarray(mesh.vertex_colors)[sortation]
        P = np.hstack((xyz, rgb))
        p_vec_gt = p_vec_gt[sortation]
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
        #render_partition_vec_o3d(mesh=P, partition=part_cp, colors=colors)
        p_cp = Partition(partition=part_cp)
        if model is None:
            _, probs_gnn, model = predict(graph_dict=graph_dict, dec_b=args.t, return_model=True, verbose=False)
        else:
            _, probs_gnn = predict(graph_dict=graph_dict, dec_b=args.t, model=model, verbose=False)
        probs_gnn = probs_gnn.reshape(probs_gnn.shape[0], )
        #print(probs_gnn.shape)
        probs_correl = predict_correl(graph_dict=graph_dict)
        probs_random = np.random.rand(probs_gnn.shape[0], )

        densities = p_gt.compute_densities(p_cp, densities_np)
        alpha = p_cp.alpha(densities)
        unions_gt = get_unions(graph_dict=graph_dict, alpha=alpha)
        probs_imperfect = get_imperfect_probs(unions=unions_gt, lam=0.1, sig=0.05)

        bce_gnn = binary_cross_entropy(y=unions_gt, probs=probs_gnn, eps=1e-6)
        bce_correl = binary_cross_entropy(y=unions_gt, probs=probs_correl, eps=1e-6)
        bce_random = binary_cross_entropy(y=unions_gt, probs=probs_random, eps=1e-6)
        bce_imperfect = binary_cross_entropy(y=unions_gt, probs=probs_imperfect, eps=1e-6)
        
        acc_gnn, prec_gnn, rec_gnn, f1_gnn, action_gnn = classification_metrics(y=unions_gt, probs=probs_gnn)
        acc_correl, prec_correl, rec_correl, f1_correl, action_correl = classification_metrics(y=unions_gt, probs=probs_correl)
        acc_random, prec_random, rec_random, f1_random, action_random = classification_metrics(y=unions_gt, probs=probs_random)
        acc_imperfect, prec_imperfect, rec_imperfect, f1_imperfect, action_imperfect = classification_metrics(y=unions_gt, probs=probs_imperfect)
        #print(acc_gnn, acc_correl, acc_random, acc_imperfect)

        exp_dict = {
            "sortation": sortation,
            "node_features": graph_dict["nodes"],
            "senders": graph_dict["senders"],
            "receivers": graph_dict["receivers"],
            "probs_gnn": probs_gnn,
            "probs_correl": probs_correl,
            "probs_random": probs_random,
            "probs_imperfect": probs_imperfect,
            "p_gt": p_vec_gt,
            "p_cp": part_cp,
            "unions_gt": unions_gt,
            "bce_gnn": np.array([bce_gnn]),
            "bce_correl": np.array([bce_correl]),
            "bce_random": np.array([bce_random]),
            "bce_imperfect": np.array([bce_imperfect]),
            "|S|": np.array([len(sp_idxs)]),
            #
            "action_gnn": action_gnn,
            "action_correl": action_correl,
            "action_random": action_random,
            "action_imperfect": action_imperfect,
            #
            "acc_gnn": np.array([acc_gnn]),
            "acc_correl": np.array([acc_correl]),
            "acc_random": np.array([acc_random]),
            "acc_imperfect": np.array([acc_imperfect]),
            #
            "prec_gnn": np.array([prec_gnn]),
            "prec_correl": np.array([prec_correl]),
            "prec_random": np.array([prec_random]),
            "prec_imperfect": np.array([prec_imperfect]),
            #
            "rec_gnn": np.array([rec_gnn]),
            "rec_correl": np.array([rec_correl]),
            "rec_random": np.array([rec_random]),
            "rec_imperfect": np.array([rec_imperfect]),
            #
            "f1_gnn": np.array([f1_gnn]),
            "f1_correl": np.array([f1_correl]),
            "f1_random": np.array([f1_random]),
            "f1_imperfect": np.array([f1_imperfect])
        }
        for j in range(len(sp_idxs)):
            sp = sp_idxs[j]
            exp_dict[str(j)] = np.array(sp)
        save_exp_data(fdir=args.h5_dir, fname=scene_name, exp_dict=exp_dict)

        """
        spc = sp_centers(sp_idxs=sp_idxs, xyz=xyz)
        render_graph(P=P, nodes=np.arange(len(sp_idxs)), sp_idxs=sp_idxs, sp_centers=spc,
            senders=graph_dict["senders"], receivers=graph_dict["receivers"], colors=colors)
        render_graph(P=P, nodes=np.arange(len(sp_idxs)), sp_idxs=sp_idxs, sp_centers=spc,
            senders=graph_dict["senders"][unions_gt], receivers=graph_dict["receivers"][unions_gt], colors=colors)
        part_cc_gt = partition(graph_dict=graph_dict, unions=unions_gt, P=P, sp_idxs=sp_idxs, half=False, verbose=verbose)
        render_partition_vec_o3d(mesh=P, partition=p_vec_gt, colors=colors)
        render_partition_vec_o3d(mesh=P, partition=part_cc_gt, colors=colors)
        """

if __name__ == "__main__":
    main()