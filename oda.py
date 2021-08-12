import numpy as np
import argparse
import os

from visu_utils import partition_pcd, initial_partition_pcd, render, pick_sp_points
from io_utils import load_cloud, save_init_graph, load_init_graph, save_probs, load_probs, save_unions, load_unions, load_colors
from ai_utils import graph, predict
from sp_utils import unify

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="File where the objects should be seperated.")
    parser.add_argument("--gpu", type=bool, default=False, help="Should the AI use the GPU. ")
    parser.add_argument("--r", type=int, default=255, help="Red channel if no color is available.")
    parser.add_argument("--g", type=int, default=0, help="Green channel if no color is available.")
    parser.add_argument("--b", type=int, default=0, help="Blue channel if no color is available.")
    parser.add_argument("--p", type=float, default=1, help="Subsampling factor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed which affects the subsampling.")
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition.")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--initial_db", default=0.5, type=float, help="Initial guess for the decision boundary.")
    parser.add_argument("--max_sp_size", default=7000, type=int, help="Maximum size of a superpoint.")
    parser.add_argument("--save_init_g", default=False, type=bool, help="Save the initial superpoint graph in g_dir.")
    parser.add_argument("--load_init_g", default=False, type=bool, help="Load the initial superpoint graph from g_dir.")
    parser.add_argument("--save_probs", default=False, type=bool, help="Save the processed superpoint graph in g_dir.")
    parser.add_argument("--load_probs", default=False, type=bool, help="Load the processed superpoint graph from g_dir.")
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--col_c", default=0.8, type=float, help="Strength of the contrast of the superpoints.")
    parser.add_argument("--load_unions", default=False, type=bool, help="Load the unions from g_dir.")
    args = parser.parse_args()
    col_c = 1 - args.col_c

    colors = load_colors()
    
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    np.random.seed(args.seed)
    P = None
    load_init_g = args.load_init_g and not args.load_probs
    if args.load_probs:
        P = -1

    if load_init_g:
        P, graph_dict, sp_idxs = load_init_graph(fdir=args.g_dir)
    if P is None:
        P = load_cloud(file=args.file, r=args.r, g=args.g, b=args.b, p=args.p)
        graph_dict, sp_idxs = graph(
            cloud=P,
            k_nn_adj=args.k_nn_adj,
            k_nn_geof=args.k_nn_geof,
            lambda_edge_weight=args.lambda_edge_weight,
            reg_strength=args.reg_strength,
            d_se_max=args.d_se_max,
            max_sp_size=args.max_sp_size)
        if args.save_init_g:
            save_init_graph(fdir=args.g_dir, P=P, graph_dict=graph_dict, sp_idxs=sp_idxs)

    if args.load_probs:
        P, graph_dict, sp_idxs, probs, unions = load_probs(fdir=args.g_dir)
    else:
        unions, probs = predict(graph_dict=graph_dict, dec_b=args.initial_db)
        if args.save_probs:
            save_probs(fdir=args.g_dir, P=P, graph_dict=graph_dict, sp_idxs=sp_idxs, probs=probs, save_init=not args.save_init_g, initial_db=args.initial_db)
    if args.load_unions:
        unions, graph_dict = load_unions(fdir=args.g_dir, graph_dict=graph_dict)

    #"""
    render(P)
    p_pcd = initial_partition_pcd(P=P, sp_idxs=sp_idxs, colors=colors)
    render(p_pcd)
    #"""
    p_pcd = partition_pcd(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, colors=colors, col_d=col_c)
    render(p_pcd)
    #"""
    if not args.load_unions:
        while True:
            d_b = input("Decision Boundary [0,1] (exit: -1):")
            try:
                d_b = float(d_b)
            except:
                continue
            if d_b == -1:
                break
            unions = np.zeros((unions.shape[0], ), dtype=np.bool)
            unions[probs > d_b] = True
            p_pcd = partition_pcd(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, colors=colors, col_d=col_c)
            render(p_pcd)
        save_unions(fdir=args.g_dir, unions=unions, graph_dict=graph_dict)
    #"""
    while True:
        picked_points_idxs = pick_sp_points(p_pcd)
        #picked_points_idxs = [8659, 54049]
        graph_dict, unions = unify(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs, graph_dict=graph_dict, unions=unions)
        p_pcd = partition_pcd(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, colors=colors, col_d=col_c)
        save_unions(fdir=args.g_dir, unions=unions, graph_dict=graph_dict)


if __name__ == "__main__":
    main()