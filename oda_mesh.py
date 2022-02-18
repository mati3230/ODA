import numpy as np
import argparse
import os
import time

from visu_utils import\
    render_o3d,\
    render_partition_vec_o3d,\
    render_partition_o3d,\
    pick_sp_points_o3d,\
    visu_dec_bs,\
    render_pptk_feat
from io_utils import\
    save_init_graph,\
    load_init_graph_mesh,\
    save_probs,\
    load_probs_mesh,\
    load_mesh,\
    save_unions,\
    load_unions,\
    save_cloud,\
    load_proc_mesh,\
    load_colors,\
    save_partition,\
    mkdir,\
    save_meshes,\
    load_binary_labels
from ai_utils import graph_mesh, predict, calculate_stris, get_delaunay_mesh
from sp_utils import\
    unify,\
    extend_superpoint,\
    separate_superpoint,\
    partition,\
    initial_partition,\
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed which affects the subsampling.")
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_adj", default=30, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--n_proc", default=10, type=int, help="Number of processes that can be used for parallel geodesic nearest neighbour search.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--initial_db", default=0.5, type=float, help="Initial guess for the decision boundary.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    parser.add_argument("--save_init_g", default=False, type=bool, help="Save the initial superpoint graph in g_dir.")
    parser.add_argument("--load_init_g", default=False, type=bool, help="Load the initial superpoint graph from g_dir.")
    parser.add_argument("--save_probs", default=False, type=bool, help="Save the processed superpoint graph in g_dir.")
    parser.add_argument("--load_probs", default=False, type=bool, help="Load the processed superpoint graph from g_dir.")
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--o_dir", default="./out", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--load_unions", default=False, type=bool, help="Load the unions from g_dir.")
    parser.add_argument("--load_proc_cloud", default=False, type=bool, help="Load the preprocessed point cloud from g_dir.")
    parser.add_argument("--ending", default="glb", type=str, help="File format of the meshes (glb, obj, ...)")
    parser.add_argument("--ignore_knn", default=False, type=bool, help="Ignore the nearest neighbour calculations to execute the l0-CP")
    parser.add_argument("--smooth", default=False, type=bool, help="Use the nearest neighbour calculations to execute the l0-CP with only the triangular graph")
    parser.add_argument("--vis_feats", default=False, type=bool, help="Only visualize the point features")
    parser.add_argument("--vis_tris", default=False, type=bool, help="Only visualize mesh and delaunay triangulation")
    args = parser.parse_args()
    if args.vis_feats:
        mesh = load_mesh(file=args.file)
        binary_labels = load_binary_labels(directory=".")
        P = np.hstack((np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors)))
        render_pptk_feat(P=P, feats=binary_labels, point_size=0.01)
        return
    if args.vis_tris:
        mesh = load_mesh(file=args.file)
        render_o3d(mesh, w_co=True)
        md = get_delaunay_mesh(mesh=mesh)
        render_o3d(md, w_co=True)
        return
    
    # load the colors for the parititon rendering
    colors = load_colors()
    colors = colors/255.


    """
    colors[0, :] = np.zeros((3,))
    colors[1, :] = np.array([1, 0, 0])
    colors[2, :] = np.array([0, 1, 0])
    colors[3, :] = np.array([0, 0, 1])
    colors[4, :] = np.array([0, 1, 1])
    """

    # Disable the tensorflow gpu computations
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    mkdir("./tmp")
    mkdir("./out")

    np.random.seed(args.seed)
    # variable to store the point cloud
    mesh = None

    # load the initial graph
    if args.load_init_g:
        mesh, graph_dict, sp_idxs, stris = load_init_graph_mesh(fdir=args.g_dir, filename=args.g_filename)
    if mesh is None: # If no initial partition is loaded, we start from the beginning by loading and preprocessing the point cloud
        if args.load_proc_cloud:
            mesh = load_proc_mesh(fdir=args.g_dir, fname=args.g_filename)
        else:
            mesh = load_mesh(file=args.file)
        editing = input("Continue Editing? [y|n]: ")
        editing = editing == "y"
        if editing:
            visualize = True
            while True:
                if visualize:
                    render_o3d(mesh, w_co=True)
                i = input("Rotate [r] | recenter [re] | Continue [-1] | Exit [e]: ")
                visualize = True
                if i == "-1":
                    break
                elif i == "e":
                    # close the application
                    return
                elif i == "r":
                    # rotate the point cloud
                    mesh = rotate(P=mesh)
                elif i == "re":
                    # translate the point cloud in the origin
                    mesh = recenter(P=mesh)
                else:
                    visualize = False
                    continue
                # the point cloud is saved after every step
                save_cloud(P=mesh, fdir=args.g_dir, fname=args.g_filename)
        
        reg_strength = args.reg_strength
        k_nn_adj = args.k_nn_adj
        lambda_edge_weight = args.lambda_edge_weight
        ignore_knn = args.ignore_knn
        smooth = args.smooth
        calc = True
        while True:            
            if calc:
                # create the initial partition graph
                # the graph_dict is used in the neural network
                # the sp_idxs is a list with a list of point indices for each superpoint
                graph_dict, sp_idxs, stris, P_f, n_sedg = \
                    graph_mesh(
                        mesh=mesh,
                        reg_strength=reg_strength,
                        lambda_edge_weight=lambda_edge_weight,
                        k_nn_adj=k_nn_adj,
                        n_proc=args.n_proc,
                        g_dir=args.g_dir,
                        g_filename=args.g_filename,
                        ignore_knn=ignore_knn,
                        smooth=smooth)
                # save the initial partition graph
                if args.save_init_g:
                    save_init_graph(
                        fdir=args.g_dir,
                        P=mesh, graph_dict=graph_dict,
                        sp_idxs=sp_idxs,
                        filename=args.g_filename,
                        stris=stris)
                init_p = initial_partition(P=mesh, sp_idxs=sp_idxs)

                #binary_labels = load_binary_labels(directory=".")
                #render_pptk_feat(P=np.asarray(mesh.vertices), feats=binary_labels)

                render_partition_o3d(mesh=mesh, sp_idxs=sp_idxs, colors=colors, w_co=True)
            calc = True
            i = input("lambda [l] | k_nn_adj [a] | lambda_edge_weight [w] | ignore knn [i] | smooth [s] | Continue [-1] | Exit [e]: ")
            if i == "-1":
                break
            elif i == "e":
                return
            elif i == "l":
                try:
                    reg_strength = float(input("lambda: "))
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "a":
                try:
                    k_nn_adj = int(input("k_nn_adj: "))
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "w":
                try:
                    lambda_edge_weight = float(input("lambda edge weight: "))
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "i":
                try:
                    ignore_knn = bool(input("ignore knn [0, 1]: "))
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            elif i == "s":
                try:
                    smooth = bool(input("smooth [0, 1]: "))
                except Exception as e:
                    print(e)
                    calc = False
                    continue
            else:
                calc = False
                continue
    # TODO continue comments here!
    if args.load_probs:
        mesh, graph_dict, sp_idxs, probs, unions, stris = load_probs_mesh(
            fdir=args.g_dir, filename=args.g_filename)
    else:
        unions, probs = predict(
            graph_dict=graph_dict, dec_b=args.initial_db, is_mesh=True)
        ######################DELETE THIS#################################
        # TODO delete the next lines and uncomment the two lines above
        #n_decisions = graph_dict["senders"].shape[0]
        #unions = np.random.randint(low=0, high=2, size=(n_decisions, )).astype(np.bool)
        #probs = np.random.rand(n_decisions, )
        ##################################################################
        if args.save_probs:
            save_probs(
                fdir=args.g_dir,
                P=mesh,
                graph_dict=graph_dict,
                sp_idxs=sp_idxs,
                probs=probs,
                save_init=not args.save_init_g,
                initial_db=args.initial_db,
                filename=args.g_filename,
                stris=stris)
    if args.load_unions:
        # TODO: bug - half is somtimes=_half 
        mesh, graph_dict, sp_idxs, stris = load_init_graph_mesh(fdir=args.g_dir, filename=args.g_filename, half="")
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

    init_p = initial_partition(P=mesh, sp_idxs=sp_idxs)
    render_partition_o3d(mesh=mesh, sp_idxs=sp_idxs, colors=colors)
    # TODO render binary labels

    #init_p = initial_partition(P=mesh, sp_idxs=sp_idxs)
    part, meshes = partition(
        graph_dict=graph_dict,
        unions=unions,
        P=mesh,
        sp_idxs=sp_idxs,
        half=False,
        stris=stris)
    save_partition(partition=part, fdir=args.g_dir, fname=args.g_filename)
    if not args.load_unions:
        pmesh = render_partition_vec_o3d(mesh=mesh, partition=part, colors=colors)
        """
        print("Nr of meshes: {0}".format(len(meshes)))
        sizes = len(meshes)*[None]
        for i in range(len(meshes)):
            mi = meshes[i]
            mv = np.asarray(mi.vertices)
            sizes[i] = mv.shape[0]
        order = np.argsort(sizes)
        selection = order[::-1]
        selection = selection[:6]
        for i in selection:
            print("mesh", i, "size", sizes[i])
            mesh_i = meshes[i]
            render_o3d(mesh_i)
        """
        while True:
            d_b = input("Decision Boundary [0,1] | Decision boundary check [d] | Continue: [-1]: ")
            if d_b == "d":
                visu_dec_bs(graph_dict=graph_dict, P=mesh, sp_idxs=sp_idxs, probs=probs, partition_func=partition)
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
            part, meshes = partition(
                graph_dict=graph_dict,
                unions=unions,
                P=mesh,
                sp_idxs=sp_idxs,
                half=False,
                stris=stris)
            save_partition(partition=part, fdir=args.g_dir, fname=args.g_filename)
            pmesh = render_partition_vec_o3d(mesh=mesh, partition=part, colors=colors)
            """
            print("Nr of meshes: {0}".format(len(meshes)))
            sizes = len(meshes)*[None]
            for i in range(len(meshes)):
                mi = meshes[i]
                mv = np.asarray(mi.vertices)
                sizes[i] = mv.shape[0]
            order = np.argsort(sizes)
            selection = order[::-1]
            selection = selection[:6]
            for i in selection:
                print("mesh", i, "size", sizes[i])
                mesh_i = meshes[i]
                render_o3d(mesh_i)
            """
        save_unions(
            fdir=args.g_dir,
            unions=unions,
            graph_dict=graph_dict,
            filename=args.g_filename)

    visualize = True
    while True:
        save_unions(
            fdir=args.g_dir,
            unions=unions,
            graph_dict=graph_dict,
            filename=args.g_filename)
        if visualize:
            pmesh = render_partition_vec_o3d(mesh=mesh, partition=part, colors=colors)
        mode = input(
            "Superpoint Editing Mode: Extend Superpoint [ex] | " +\
            "Create [c] | " +\
            "Separate [s] | " +\
            "Extend points [ep] | " +\
            "Reduce points [r] | " +\
            "Merge small ones [m] | " +\
            "Unify [u] | " +\
            "Superpoint info [i] | " +\
            "Exit [e]: ")
        visualize = True
        if mode == "e":
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
            unions = merge_small_singles(
                graph_dict=graph_dict, sp_idxs=sp_idxs,
                unions=unions, thres=thres, P=mesh)
        if mode != "c" and\
                mode != "ex" and\
                mode != "s" and\
                mode != "ep" and\
                mode != "r" and\
                mode != "i" and\
                mode != "m" and\
                mode != "u":
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
                print("Pick two superpoints that will be unified to a new superpoint")
            elif mode == "s":
                print("Select a superpoint were the unions should be deleted. This is a seperation of a superpoint.")
            elif mode == "ep":
                print("Pick a superpoint that should be extended with points of the cloud.")
            elif mode == "r":
                print("Pick a superpoint where you want to reduce points.")
            picked_points_idxs = pick_sp_points_o3d(pcd=pmesh, is_mesh=True)
        if mode == "c":
            # unify superpoints
            graph_dict, unions = unify(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs,
                graph_dict=graph_dict,
                unions=unions)
        elif mode == "ex":
            print("Select multiple superpoints. The chosen superpoint will be extended with this points.")
            points_idxs_e = pick_sp_points_o3d(pcd=pmesh, is_mesh=True)
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
            mesh_sp = superpoint_info(
                picked_points_idxs=picked_points_idxs,
                sp_idxs=sp_idxs, P=mesh, graph_dict=graph_dict, stris=stris)
            render_o3d(mesh_sp)
        elif mode == "ep":
            print("Pick points that should be added to the superpoint chosen. ")
            points_idxs_e = pick_sp_points_o3d(pcd=pmesh, is_mesh=True)
            sp_idxs = extend_superpoint_points(picked_points_idxs=picked_points_idxs, points_idxs_e=points_idxs_e, sp_idxs=sp_idxs)
            init_p = initial_partition(P=mesh, sp_idxs=sp_idxs)
            stris, _, _, _ =\
                calculate_stris(
                    tris=np.asarray(mesh.triangles),
                    partition_vec=init_p, sp_idxs=sp_idxs)
        elif mode == "r":
            print("Pick points that should be reduced from the superpoint chosen. ")
            points_idxs_r = pick_sp_points_o3d(pcd=pmesh, is_mesh=True)
            graph_dict, sp_idxs = reduce_superpoint(
                picked_points_idxs=picked_points_idxs,
                points_idxs_r=points_idxs_r, graph_dict=graph_dict,
                sp_idxs=sp_idxs)
            init_p = initial_partition(P=mesh, sp_idxs=sp_idxs)
            stris, _, _, _ =\
                calculate_stris(
                    tris=np.asarray(mesh.triangles),
                    partition_vec=init_p, sp_idxs=sp_idxs)
        
        if args.save_init_g:
            save_init_graph(
                fdir=args.g_dir,
                P=mesh, graph_dict=graph_dict,
                sp_idxs=sp_idxs,
                filename=args.g_filename,
                stris=stris,
                half="_half")
        part, meshes = partition(
            graph_dict=graph_dict,
            unions=unions,
            P=mesh,
            sp_idxs=sp_idxs,
            half=False,
            stris=stris)
        save_partition(partition=part, fdir=args.g_dir, fname=args.g_filename)
        save_meshes(meshes=meshes, fdir=args.o_dir, filename=args.g_filename, ending=args.ending)


if __name__ == "__main__":
    main()