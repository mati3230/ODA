import numpy as np
import argparse
import os
import time

from visu_utils import\
    render_pptk,\
    pick_sp_points_pptk,\
    visu_dec_bs
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
    subsample,\
    save_partition,\
    mkdir
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
    recenter,\
    merge_small_singles,\
    superpoint_info,\
    unify_superpoints


def main():
    """
    This is the app that creates partitions from a point cloud.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="File where the objects should be seperated.")
    parser.add_argument("--gpu", type=bool, default=False, help="Should the AI use the GPU. ")
    parser.add_argument("--r", type=int, default=255, help="Red channel if no color is available.")
    parser.add_argument("--g", type=int, default=0, help="Green channel if no color is available.")
    parser.add_argument("--b", type=int, default=0, help="Blue channel if no color is available.")
    parser.add_argument("--p", type=float, default=1, help="Subsampling factor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed which affects the subsampling.")
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--initial_db", default=0.5, type=float, help="Initial guess for the decision boundary.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    parser.add_argument("--save_init_g", default=False, type=bool, help="Save the initial superpoint graph in g_dir.")
    parser.add_argument("--load_init_g", default=False, type=bool, help="Load the initial superpoint graph from g_dir.")
    parser.add_argument("--save_probs", default=False, type=bool, help="Save the processed superpoint graph in g_dir.")
    parser.add_argument("--load_probs", default=False, type=bool, help="Load the processed superpoint graph from g_dir.")
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--load_unions", default=False, type=bool, help="Load the unions from g_dir.")
    parser.add_argument("--load_proc_cloud", default=False, type=bool, help="Load the preprocessed point cloud from g_dir.")
    parser.add_argument("--partition_eval", default=False, type=bool, help="Evaluate the partition algorithm.")
    parser.add_argument("--point_size", default=0.03, type=float, help="Rendering point size.")
    args = parser.parse_args()
    # load the colors for the parititon rendering
    colors = load_colors()
    colors = colors/255.
    viewer = None

    # Disable the tensorflow gpu computations
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    mkdir("./tmp")
    mkdir("./out")
    np.random.seed(args.seed)
    # variable to store the point cloud
    P = None
    # should the initial partition be loaded? 
    # not necessary if the model probabilities of the binary link prediction should be loaded
    load_init_g = args.load_init_g and not args.load_probs
    if args.load_probs:
        P = -1

    # load the initial graph
    if load_init_g:
        P, graph_dict, sp_idxs = load_init_graph(fdir=args.g_dir, filename=args.g_filename)
    if P is None: # If no initial partition is loaded, we start from the beginning by loading and preprocessing the point cloud
        if args.load_proc_cloud:
            P = load_proc_cloud(fdir=args.g_dir, fname=args.g_filename)
        else:
            P = load_cloud(file=args.file, r=args.r, g=args.g, b=args.b, p=args.p)
        editing = input("Continue Editing? [y|n]: ")
        editing = editing == "y"
        if editing:
            visualize = True
            while True:
                if visualize:
                    viewer = render_pptk(P=P, v=viewer)
                i = input("Delete points [d] | Rotate [r] | recenter [re] | downsample [s] | Continue [-1] | Exit [e]: ")
                visualize = True
                if i == "-1":
                    break
                elif i == "e":
                    # close the application
                    if viewer is not None:
                        viewer.close()
                    return
                elif i == "d":
                    # remove the selection from the point cloud 
                    picked_points_idxs, viewer = pick_sp_points_pptk(
                        P=P, point_size=args.point_size, v=viewer, colors=colors)
                    P = delete(P=P, idxs=picked_points_idxs)
                elif i == "r":
                    # rotate the point cloud
                    P = rotate(P=P)
                elif i == "re":
                    # translate the point cloud in the origin
                    P = recenter(P=P)
                elif i == "s":
                    # downsample the point cloud
                    print("Point cloud size: {0}".format(P.shape[0]))
                    s = input("Percent [0,1]: ")
                    try:
                        s = float(s)
                        P = subsample(P=P, p=s)
                    except Exception as e:
                        print(e)
                        continue
                else:
                    visualize = False
                    continue
                # the point cloud is saved after every step
                save_cloud(P=P, fdir=args.g_dir, fname=args.g_filename)
        #return
        if args.partition_eval:
            print("Start evaluation")
            eval_dir = "./eval"
            mkdir(eval_dir)
            reg_strengths = [0.05, 0.1, 0.3, 0.7]
            ei = 1
            for rs in reg_strengths:
                start_time = time.time()
                graph_dict, sp_idxs = graph(
                    cloud=P,
                    k_nn_adj=args.k_nn_adj,
                    k_nn_geof=args.k_nn_geof,
                    lambda_edge_weight=args.lambda_edge_weight,
                    reg_strength=rs,
                    d_se_max=args.d_se_max,
                    max_sp_size=args.max_sp_size)
                stop_time = time.time()
                duration = stop_time - start_time
                unions, probs = predict(graph_dict=graph_dict, dec_b=args.initial_db)
                senders = graph_dict["senders"]
                if 2*unions.shape[0] == senders.shape[0]:
                    half = int(senders.shape[0] / 2)
                    senders = senders[:half]
                    receivers = graph_dict["receivers"]
                    receivers = receivers[:half]
                    graph_dict["senders"] = senders
                    graph_dict["receivers"] = receivers
                for db in np.arange(0.8, 1, 0.01):
                    unions = np.zeros((unions.shape[0], ), dtype=np.bool)
                    unions[probs > db] = True
                    part = partition(
                        graph_dict=graph_dict,
                        unions=unions,
                        P=P,
                        sp_idxs=sp_idxs,
                        half=False)
                    save_partition(partition=part, fdir=eval_dir, fname=str(ei)+"_{0:.5f}".format(duration))
                    ei += 1
            return

        reg_strength = args.reg_strength
        k_nn_adj = args.k_nn_adj
        k_nn_geof = args.k_nn_geof
        lambda_edge_weight = args.lambda_edge_weight
        calc = True
        while True:            
            if calc:
                # create the initial partition graph
                # the graph_dict is used in the neural network
                # the sp_idxs is a list with a list of point indices for each superpoint
                graph_dict, sp_idxs = graph(
                    cloud=P,
                    k_nn_adj=args.k_nn_adj,
                    k_nn_geof=args.k_nn_geof,
                    lambda_edge_weight=args.lambda_edge_weight,
                    reg_strength=args.reg_strength,
                    d_se_max=args.d_se_max,
                    max_sp_size=args.max_sp_size)
                # save the initial partition graph
                if args.save_init_g:
                    save_init_graph(
                        fdir=args.g_dir,
                        P=P, graph_dict=graph_dict,
                        sp_idxs=sp_idxs,
                        filename=args.g_filename)
            calc = True
            i = input("lambda [l] | k_nn_adj [a] | k_nn_geof[g] | lambda_edge_weight [w] | Continue [-1] | Exit [e]: ")
            if i == "-1":
                break
            elif i == "e":
                return
            elif i == "l":
                try:
                    reg_strength = float(i)
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "a":
                try:
                    k_nn_adj = int(i)
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "g":
                try:
                    k_nn_geof = int(i)
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "w":
                try:
                    lambda_edge_weight = float(i)
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            else:
                calc = False
                continue
    # TODO continue comments here!
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
        # TODO: bug - half is somtimes=_half 
        P, graph_dict, sp_idxs = load_init_graph(fdir=args.g_dir, filename=args.g_filename, half="")
        unions, graph_dict = load_unions(
            fdir=args.g_dir, graph_dict=graph_dict, filename=args.g_filename)
    senders = graph_dict["senders"]
    if 2*unions.shape[0] == senders.shape[0]:
        half = int(senders.shape[0] / 2)
        senders = senders[:half]
        receivers = graph_dict["receivers"]
        receivers = receivers[:half]
        graph_dict["senders"] = senders
        graph_dict["receivers"] = receivers

    init_p = initial_partition(P=P, sp_idxs=sp_idxs)
    part = partition(
        graph_dict=graph_dict,
        unions=unions,
        P=P,
        sp_idxs=sp_idxs,
        half=False)
    save_partition(partition=part, fdir=args.g_dir, fname=args.g_filename)
    print("You can switch between the colored point cloud, the initial partition and the GNN partition by pressing: AltGr + ]")
    if not args.load_unions:
        viewer = render_pptk(P=P, initial_partition=init_p, partition=part, point_size=args.point_size, v=viewer, colors=colors)
        while True:
            d_b = input("Decision Boundary [0,1] | Decision boundary check [d] | Continue: [-1]: ")
            if d_b == "d":
                visu_dec_bs(graph_dict=graph_dict, P=P, sp_idxs=sp_idxs, probs=probs, partition_func=partition)
                continue
            try:
                d_b = float(d_b)
            except:
                continue
            if d_b == -1:
                break
            unions = np.zeros((unions.shape[0], ), dtype=np.bool)
            t1 = time.time()
            unions[probs > d_b] = True
            t2 = time.time()
            duration = t2 - t1
            print("Threshold operation takes {0:.3f} seconds.".format(duration))
            part = partition(
                graph_dict=graph_dict,
                unions=unions,
                P=P,
                sp_idxs=sp_idxs,
                half=False)
            save_partition(partition=part, fdir=args.g_dir, fname=args.g_filename)
            viewer = render_pptk(P=P, initial_partition=init_p, partition=part, point_size=args.point_size, v=viewer, colors=colors)
        save_unions(
            fdir=args.g_dir,
            unions=unions,
            graph_dict=graph_dict,
            filename=args.g_filename)

    point_size = args.point_size
    visualize = True
    while True:
        save_unions(
            fdir=args.g_dir,
            unions=unions,
            graph_dict=graph_dict,
            filename=args.g_filename)
        if visualize:
            viewer = render_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
        mode = input("Superpoint Editing Mode: Extend Superpoint [ex] | Create [c] | Separate [s] | Point_Size [ps] | Extend points [ep] | Reduce points [r] | Merge small ones [m] | Unify [u] | Superpoint info [i] | Exit [e]: ")
        visualize = True
        if mode == "ps":
            ps = input("Rendering point size: ")
            try:
                ps = float(ps)
            except:
                continue
            point_size = ps
            continue
        elif mode == "e":
            if viewer is not None:
                viewer.close()
            return
        elif mode == "m":
            print("Merge superpoints that are smaller than a certain threshold.")
            thres = input("Minimum superpoint size: ")
            try:
                thres = int(thres)
            except:
                continue
            if thres <= 0:
                continue
            unions = merge_small_singles(graph_dict=graph_dict, sp_idxs=sp_idxs, unions=unions, thres=thres, P=P)
        if mode != "c" and mode != "ex" and mode != "s" and mode != "ep" and mode != "r" and mode != "i" and mode != "m" and mode != "u":
            continue
        pick_points = True
        if mode == "m":
            pick_points = False
        if pick_points:
            if mode == "c":
                print("Select superpoints that will be merged.")
            elif mode == "ex":
                print("Select one superpoint that should be extended.")
            elif mode == "u":
                print("Pick some points that will be unified to a new superpoint")
            elif mode == "s":
                print("Select a superpoint were the unions should be deleted. This is a seperation of a superpoint.")
            elif mode == "ep":
                print("Pick a superpoint that should be extended with points of the cloud.")
            elif mode == "r":
                print("Pick a superpoint where you want to reduce points.")
            picked_points_idxs, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
        if mode == "c":
            graph_dict, unions = unify(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "ex":
            print("Select multiple superpoints. The chosen superpoint will be extended with this points.")
            points_idxs_e, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            graph_dict, unions = extend_superpoint(
                picked_points_idxs=picked_points_idxs,
                points_idxs_e=points_idxs_e,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "u":
            # picked_points_idxs, sp_idxs, graph_dict, unions
            unions = unify_superpoints(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "s":
            graph_dict, unions = separate_superpoint(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "i":
            P_sp = superpoint_info(picked_points_idxs=picked_points_idxs, sp_idxs=sp_idxs, P=P, graph_dict=graph_dict)
            viewer = render_pptk(P=P_sp, v=viewer)
        elif mode == "ep":
            print("Pick points that should be added to the superpoint chosen. ")
            points_idxs_e, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            sp_idxs = extend_superpoint_points(picked_points_idxs=picked_points_idxs, points_idxs_e=points_idxs_e, sp_idxs=sp_idxs)
        elif mode == "r":
            print("Pick points that should be reduced from the superpoint chosen. ")
            points_idxs_r, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size, v=viewer, colors=colors)
            graph_dict, sp_idxs = reduce_superpoint(picked_points_idxs=picked_points_idxs, points_idxs_r=points_idxs_r, graph_dict=graph_dict, sp_idxs=sp_idxs)
            init_p = initial_partition(P=P, sp_idxs=sp_idxs)
        if args.save_init_g:
            save_init_graph(
                fdir=args.g_dir,
                P=P, graph_dict=graph_dict,
                sp_idxs=sp_idxs,
                filename=args.g_filename,
                half="_half")
        part = partition(
            graph_dict=graph_dict,
            unions=unions,
            P=P,
            sp_idxs=sp_idxs,
            half=False)
        save_partition(partition=part, fdir=args.g_dir, fname=args.g_filename)


if __name__ == "__main__":
    main()