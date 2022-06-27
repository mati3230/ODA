import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scannet_utils import get_scenes, get_ground_truth
from ai_utils import graph, predict
from partition.felzenszwalb import partition_from_probs
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


def get_csv_header(algorithms=["FH04", "CP"]):
    header = [
        "Name",
        "K"
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
    parser.add_argument("--min_k", type=float, default=1000, help="Minimum k parameter of FH04 segmentation algorithm.")
    parser.add_argument("--max_k", type=float, default=100, help="Maximum k parameter of FH04 segmentation algorithm.")
    parser.add_argument("--step_k", type=float, default=100, help="Step size to which the k parameter of FH04 segmentation algorithm will be varied.")
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    #
    #parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--csv_dir", default="./", type=str, help="Directory where we save the csv.")
    #
    parser.add_argument("--sn_id", default=0, type=int, help="Scannet scene id in the range of [0,707]")
    args = parser.parse_args()

    if args.min_k >= args.max_k:
        raise Exception("min_k {0} should be smaller than max_k {1}".format(args.min_k, args.max_k))
    if args.step_k >= args.max_k:
        raise Exception("step_k {0} should be smaller than max_k {1}".format(args.step_k, args.max_k))
    k_params = np.arange(start=args.min_k, stop=args.max_k, step=args.step_k)

    max_sn_id = 707
    min_sn_id = 0
    if args.sn_id > max_sn_id or args.sn_id < min_sn_id:
        raise Exception("Scannet ID {0} should be between {1} and {2}.".format(args.sn_id, min_sn_id, max_sn_id))
    scenes, _, scannet_dir = get_scenes(blacklist=[])
    scene_name = scenes[args.sn_id]

    csv_header = get_csv_header()
    csv_name = "fh04_k_id{0}".format(args.sn_id)
    
    verbose = False
    model = None

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

    unions, probs, model = predict(graph_dict=graph_dict, dec_b=0.8, return_model=True, verbose=False)

    ooas = []
    sizes = []
    all_data = []
    partition_gt = None
    sortation = None

    ooa_cp, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_cp, partition_gt=partition_gt, sortation=sortation)
    size_cp = len(sp_idxs)

    for i in tqdm(range(k_params.shape[0]), desc="Vary K of FH04", disable=verbose):
        k = k_params[i]
        part_fh04 = partition_from_probs(graph_dict=graph_dict, sim_probs=probs, k=k, P=P, sp_idxs=sp_idxs)
        ooa_fh04, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_fh04, partition_gt=partition_gt, sortation=sortation)
        size_fh04 = np.unique(part_fh04).shape[0]
        ooas.append(ooa_fh04)
        sizes.append(size_fh04)
        all_data.append([scene_name, k, ooa_fh04, size_fh04, ooa_cp, size_cp])
    
    save_csv(res=all_data, csv_dir=args.csv_dir, csv_name=csv_name, data_header=csv_header)

    ooas = np.array(ooas)
    sizes = np.array(sizes)

    ooa_cps = np.zeros((ooas.shape[0], ))
    ooa_cps[:] = ooa_cp

    plt.plot(k_params, ooas, ".")
    plt.plot(k_params, ooa_cps, "-")
    #plt.xticks(k_params)
    plt.savefig(fname="fh04_k_ooa_id{0}.jpg".format(args.sn_id))
    plt.show()


    sizes_cps = np.zeros((sizes.shape[0], ))
    sizes_cps[:] = size_cp
    plt.plot(k_params, sizes, ".")
    plt.plot(k_params, sizes_cps, "-")
    #plt.xticks(k_params)
    plt.savefig(fname="fh04_k_sizes_id{0}.jpg".format(args.sn_id))
    plt.show()


def all_scenes():
    parser = argparse.ArgumentParser()
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    #
    parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--csv_dir", default="./", type=str, help="Directory where we save the csv.")
    args = parser.parse_args()

    k_params = [0.001, 0.1, 1, 2, 3, 4, 10]

    scenes, _, scannet_dir = get_scenes(blacklist=[])
    
    csv_header = get_csv_header()
    csv_name = "fh04_k_all"
    
    verbose = False
    model = None

    for j in range(len(scenes)):
        scene_name = scenes[j]
        mesh, p_vec_gt, file_gt = get_ground_truth(
            scannet_dir=scannet_dir, scene=scene_name)
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

        unions, probs, model = predict(graph_dict=graph_dict, dec_b=0.8, return_model=True, verbose=False)

        ooas = []
        sizes = []
        all_data = []
        partition_gt = None
        sortation = None

        ooa_cp, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_cp, partition_gt=partition_gt, sortation=sortation)
        size_cp = len(sp_idxs)

        for i in tqdm(range(len(k_params)), desc="Vary K of FH04", disable=verbose):
            k = k_params[i]
            part_fh04 = partition_from_probs(graph_dict=graph_dict, sim_probs=probs, k=k, P=P, sp_idxs=sp_idxs)
            ooa_fh04, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=part_fh04, partition_gt=partition_gt, sortation=sortation)
            size_fh04 = np.unique(part_fh04).shape[0]
            ooas.append(ooa_fh04)
            sizes.append(size_fh04)
            all_data.append([scene_name, k, ooa_fh04, size_fh04, ooa_cp, size_cp])
        if j % args.pkg_size == 0 and len(csv_data) > 0:
            save_csv(res=all_data, csv_dir=args.csv_dir, csv_name=str(i), data_header=csv_header)
            csv_data.clear()


if __name__ == "__main__":
    #main()
    all_scenes()