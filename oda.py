import numpy as np
import open3d as o3d
import argparse
import os

from visu_utils import visualizer, partition_pcd
from io_utils import load
from ai_utils import graph, predict

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
    args = parser.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    np.random.seed(args.seed)
    P, pcd = load(file=args.file, r=args.r, g=args.g, b=args.b, p=args.p)
    """
    n_P = 10000
    P = 5*np.random.randn(n_P, 3)
    C = np.random.randint(low=0, high=255, size=(n_P, 3))
    P = np.hstack((P, C)).astype(np.float32)
    """
    graph_dict, sp_idxs = graph(
        P=P,
        k_nn_adj=args.k_nn_adj,
        k_nn_geof=args.k_nn_geof,
        lambda_edge_weight=args.lambda_edge_weight,
        reg_strength=args.reg_strength,
        d_se_max=args.d_se_max)
    unions, probs = predict(graph_dict=graph_dict)

    data = np.load("colors.npz")
    colors = data["colors"]
    p_pcd = partition_pcd(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, colors=colors)
    o3d.visualization.draw_geometries([p_pcd])
    #vis = visualizer()
    #vis.add_geometry(p_pcd)
    #vis.run()
    #vis.destroy_window()
    while True:
        d_b = input("Decision Boundary [0,1] (exit: -1):")
        try:
            d_b = float(d_b)
        except:
            continue
        if d_b == -1:
            return
        unions = np.zeros((unions.shape[0], ), dtype=np.bool)
        unions[probs > d_b] = True
        p_pcd = partition_pcd(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, colors=colors)
        o3d.visualization.draw_geometries([p_pcd])


if __name__ == "__main__":
    main()