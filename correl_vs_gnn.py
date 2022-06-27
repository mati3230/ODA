import argparse
import os
from tqdm import tqdm
import numpy as np

"""
from scannet_utils import get_scenes, get_ground_truth
from ai_utils import graph, predict, np_fast_dot
from partition.felzenszwalb import partition_from_probs
from partition.partition import Partition
from partition.density_utils import densities_np
"""

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


def get_csv_header(algorithms=["GNN", "Correl"]):
    header = [
        "Name",
        "|P|"
    ]
    algo_stats = [
        "OOA",
        "|S|",
        "BCE"
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


def get_unions(graph_dict, alpha):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]

    unions = np.zeros((senders.shape[0], ), dtype=np.bool)
    for i in range(senders.shape[0]):
        e_i = senders[i]
        e_j = receivers[j]
        unions[i] = alpha[e_i] == alpha[e_j]
    return unions


def predict_correl(graph_dict):
    node_features = graph_dict["nodes"]
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]

    probs = np.zeros((senders.shape[0], ), dtype=np.float32)
    for i in range(senders.shape[0]):
        e_i = senders[i]
        e_j = receivers[j]
        f_i = node_features[e_i]
        f_j = node_features[e_j]
        correlation, _, _ = np_fast_dot(a=f_i, b=f_j)
        probs[i] = (correlation + 1) / 2
    return probs


def binary_cross_entropy(y, probs, eps=1e-6):
    y = y.astype(np.float32)
    zero_idxs = np.where(probs == 0)[0]
    probs[zero_idxs] = eps
    pos_bce = -y * np.log(probs)

    zero_idxs = np.where((1-probs) == 0)[0]
    probs[zero_idxs] = 1-eps
    neg_bce = -(1-y) * np.log(1-probs)

    bce = pos_bce + neg_bce
    bce = np.mean(bce)
    return bce


def test_bce():
    unions_gt = np.random.rand(100, 1)
    unions_gt = unions_gt.reshape(unions_gt.shape[0], )
    unions_gt = unions_gt > 0.5
    #print(unions_gt.astype(np.float32))
    #probs = np.random.rand(100, 1)
    #probs = probs.reshape(probs.shape[0], )
    probs = np.array(unions_gt, copy=True)
    probs = probs.astype(np.float32)
    deviation = 0.01
    probs[unions_gt == 1] -= deviation # deviation from 1
    probs[unions_gt == 0] += deviation # deviation from 0
    bce = binary_cross_entropy(y=unions_gt, probs=probs)
    print("BCE={0}".format(bce))


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
    
    csv_header = get_csv_header()
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
        bce_correl = binary_cross_entropy(y=unions_gt, probs=probs_gnn)

        csv_data.append([file_gt, P.shape[0], ooa_gnn, size_gnn, bce_gnn, ooa_correl, size_correl, bce_correl])
        if i % args.pkg_size == 0 and len(csv_data) > 0:
            save_csv(res=csv_data, csv_dir=args.csv_dir, csv_name=str(i), data_header=csv_header)
            csv_data.clear()

        
if __name__ == "__main__":
    main()
    #test_bce()