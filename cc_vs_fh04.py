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


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def write_csv(filedir, filename, csv):
    if filedir[-1] != "/":
        filedir += "/"
    if not filename.endswith(".csv"):
        filename += ".csv"
    out_filename = filedir + filename
    file = open(out_filename, "w")
    file.write(csv)
    file.close()


def save_csv(res, csv_dir, csv_name, data_header):
    # write results of the cut pursuit calculations as csv
    csv = ""
    for header in data_header:
        csv += header + ","
    csv = csv[:-1] + "\n"

    for tup in res:
        if tup is None:
            continue
        for elem in tup:
            if type(elem) == list:
                csv += str(elem)[1:-1].replace(" ", "") + ","
            else:
                csv += str(elem) + ","
        csv = csv[:-1] + "\n"
    write_csv(filedir=csv_dir, filename=csv_name, csv=csv)


def get_csv_header(algorithms=["FH04", "CC", "CP"]):
    header = [
        "Name",
        "|P|"
    ]
    algo_stats = [
        "OOA",
        "|S|"
    ]

    for algo in algorithms:
        for algs in algo_stats:
            header.append(algs + "_" + algo)
    return header


def ooa(par_v_gt, par_v, partition_gt=None, sortation=None):
    precalc = partition_gt is not None and sortation is not None
    if precalc:
        par_v = par_v[sortation]
        partition_A = Partition(partition=par_v)
    else:
        sortation = np.argsort(par_v_gt)
        par_v_gt = par_v_gt[sortation]
        par_v = par_v[sortation]
        
        ugt, ugt_idxs, ugt_counts = np.unique(par_v_gt, return_index=True, return_counts=True)

        partition_gt = Partition(partition=par_v_gt, uni=ugt, idxs=ugt_idxs, counts=ugt_counts)
        partition_A = Partition(partition=par_v)
        
    max_density = partition_gt.partition.shape[0]
    ooa_A, _ = partition_gt.overall_obj_acc(max_density=max_density, partition_B=partition_A, density_function=densities_np)
    return ooa_A, partition_gt, sortation


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
    args = parser.parse_args()

    mkdir(args.csv_dir)

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    n_scenes = len(scenes)
    
    csv_header = get_csv_header()
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