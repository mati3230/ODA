import numpy as np
import argparse
import os

from visu_utils import\
    render_pptk,\
    pick_sp_points_pptk
from io_utils import\
    load_cloud,\
    save_init_graph,\
    load_init_graph,\
    save_probs,\
    load_probs,\
    save_unions,\
    load_unions,\
    save_cloud,\
    load_proc_cloud,\
    load_colors,\
    subsample
from ai_utils import graph, predict
from sp_utils import\
    unify,\
    extend_superpoint,\
    separate_superpoint,\
    partition,\
    initial_partition,\
    delete,\
    extend_superpoint_points,\
    reduce_superpoint,\
    rotate,\
    recenter

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
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--load_unions", default=False, type=bool, help="Load the unions from g_dir.")
    parser.add_argument("--load_proc_cloud", default=False, type=bool, help="Load the preprocessed point cloud from g_dir.")
    parser.add_argument("--point_size", default=0.03, type=float, help="Rendering point size.")
    args = parser.parse_args()
    colors = load_colors()
    colors = colors/255.
    viewer = None
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    np.random.seed(args.seed)
    P = None
    load_init_g = args.load_init_g and not args.load_probs
    if args.load_probs:
        P = -1

    if load_init_g:
        P, graph_dict, sp_idxs = load_init_graph(fdir=args.g_dir, filename=args.g_filename)
    if P is None:
        if args.load_proc_cloud:
            P = load_proc_cloud(fdir=args.g_dir, fname=args.g_filename)
        else:
            P = load_cloud(file=args.file, r=args.r, g=args.g, b=args.b, p=args.p)
        editing = input("Continue Editing? [y|n]: ")
        editing = editing == "y"
        if editing:
            while True:
                viewer = render_pptk(P=P, v=viewer)
                i = input("Delete points [d] | Rotate [r] | recenter [re] | downsample [s] | Continue [-1] | Exit [e]: ")
                if i == "-1":
                    break
                elif i == "e":
                    if viewer is not None:
                        viewer.close()
                    return
                elif i == "d":
                    picked_points_idxs, viewer = pick_sp_points_pptk(
                        P=P, point_size=args.point_size, v=viewer, colors=colors)
                    P = delete(P=P, idxs=picked_points_idxs)
                elif i == "r":
                    P = rotate(P=P)
                elif i == "re":
                    P = recenter(P=P)
                elif i == "s":
                    s = input("Percent [0,1]: ")
                    try:
                        s = float(s)
                        P = subsample(P=P, p=s)
                    except Exception as e:
                        print(e)
                        continue
                else:
                    continue
                save_cloud(P=P, fdir=args.g_dir, fname=args.g_filename)
        #return
        graph_dict, sp_idxs = graph(
            cloud=P,
            k_nn_adj=args.k_nn_adj,
            k_nn_geof=args.k_nn_geof,
            lambda_edge_weight=args.lambda_edge_weight,
            reg_strength=args.reg_strength,
            d_se_max=args.d_se_max,
            max_sp_size=args.max_sp_size)
        if args.save_init_g:
            save_init_graph(
                fdir=args.g_dir,
                P=P, graph_dict=graph_dict,
                sp_idxs=sp_idxs,
                filename=args.g_filename)

    if args.load_probs:
        P, graph_dict, sp_idxs, probs, unions = load_probs(
            fdir=args.g_dir, filename=args.g_filename)
    else:
        unions, probs = predict(graph_dict=graph_dict, dec_b=args.initial_db)
        if args.save_probs:
            save_probs(
                fdir=args.g_dir,
                P=P,
                graph_dict=graph_dict,
                sp_idxs=sp_idxs,
                probs=probs,
                save_init=not args.save_init_g,
                initial_db=args.initial_db,
                filename=args.g_filename)
    if args.load_unions:
        unions, graph_dict = load_unions(
            fdir=args.g_dir, graph_dict=graph_dict, filename=args.g_filename)
    init_p = initial_partition(P=P, sp_idxs=sp_idxs)
    part = partition(
        graph_dict=graph_dict,
        unions=unions,
        P=P,
        sp_idxs=sp_idxs)
    viewer = render_pptk(P=P, initial_partition=init_p, partition=part, point_size=args.point_size, v=viewer, colors=colors)
    #"""
    if not args.load_unions:
        while True:
            d_b = input("Decision Boundary [0,1] | Continue: [-1]: ")
            try:
                d_b = float(d_b)
            except:
                continue
            if d_b == -1:
                break
            unions = np.zeros((unions.shape[0], ), dtype=np.bool)
            unions[probs > d_b] = True
            part = partition(
                graph_dict=graph_dict,
                unions=unions,
                P=P,
                sp_idxs=sp_idxs)
            viewer = render_pptk(P=P, initial_partition=init_p, partition=part, point_size=args.point_size, v=viewer, colors=colors)
        save_unions(
            fdir=args.g_dir,
            unions=unions,
            graph_dict=graph_dict,
            filename=args.g_filename)
    #"""
    point_size = args.point_size
    while True:
        save_unions(
            fdir=args.g_dir,
            unions=unions,
            graph_dict=graph_dict,
            filename=args.g_filename)
        mode = input("Superpoint Editing Mode: Extend [ex] | Create [c] | Separate [s] | Point Cloud [p] | Point_Size [ps] | Extend points [ep] | Reduce points [r] | Exit [e]: ")
        if mode == "p":
            viewer = render_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            continue
        elif mode == "ps":
            ps = input("Point size: ")
            try:
                ps = float(ps)
            except:
                continue
            point_size = ps
            continue
        elif mode == "-1":
            if viewer is not None:
                viewer.close()
            return
        picked_points_idxs, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
        if mode == "c":
            graph_dict, unions = unify(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "e":
            points_idxs_e, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            graph_dict, unions = extend_superpoint(
                picked_points_idxs=picked_points_idxs,
                points_idxs_e=points_idxs_e,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "s":
            graph_dict, unions = separate_superpoint(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "ep":
            points_idxs_e, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            sp_idxs = extend_superpoint_points(picked_points_idxs=picked_points_idxs, points_idxs_e=points_idxs_e, sp_idxs=sp_idxs)
        elif mode == "r":
            points_idxs_r, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            graph_dict, sp_idxs = reduce_superpoint(picked_points_idxs=picked_points_idxs, points_idxs_r=points_idxs_r, graph_dict=graph_dict, sp_idxs=sp_idxs)
            init_p = initial_partition(P=P, sp_idxs=sp_idxs)
            if args.save_init_g:
                save_init_graph(
                    fdir=args.g_dir,
                    P=P, graph_dict=graph_dict,
                    sp_idxs=sp_idxs,
                    filename=args.g_filename)
        part = partition(
            graph_dict=graph_dict,
            unions=unions,
            P=P,
            sp_idxs=sp_idxs)


if __name__ == "__main__":
    main()