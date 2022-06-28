import os
import numpy as np

#"""
from ai_utils import np_fast_dot
from partition.partition import Partition
from partition.density_utils import densities_np
import h5py
#"""


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


def get_csv_header(header=["Name", "|P|"], algorithms=["GNN", "Correl"], algo_stats=["OOA", "|S|", "BCE"]):
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
        e_j = receivers[i]
        unions[i] = alpha[e_i] == alpha[e_j]
    return unions


def predict_correl(graph_dict):
    node_features = graph_dict["nodes"]
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]

    probs = np.zeros((senders.shape[0], ), dtype=np.float32)
    for i in range(senders.shape[0]):
        e_i = senders[i]
        e_j = receivers[i]
        f_i = node_features[e_i]
        f_j = node_features[e_j]
        correlation, _, _ = np_fast_dot(a=f_i, b=f_j)
        probs[i] = (correlation + 1) / 2
    return probs


def binary_cross_entropy(y, probs, eps=1e-6):
    y = y.astype(np.float32)
    zero_idxs = np.where(probs == 0)[0]
    probs[zero_idxs] = eps
    pos_idxs = np.where(y == 1)[0]
    pos_bce = -np.log(probs[pos_idxs])

    zero_idxs = np.where((1-probs) == 0)[0]
    probs[zero_idxs] = 1-eps
    neg_idxs = np.where(y == 0)[0]
    neg_bce = -np.log(1-probs[neg_idxs])

    bce = pos_bce + neg_bce
    bce = np.mean(bce)
    return bce


def get_imperfect_probs(unions, lam=0.1, sig=0.05):
    u = np.array(unions, copy=True)
    u = u.astype(np.float32)
    pos_idxs = np.where(u == 1)[0]
    neg_idxs = np.where(u == 0)[0]
    u[pos_idxs] -= lam * np.random.normal(loc=0, scale=sig, size=(pos_idxs.shape[0], ))
    u[neg_idxs] += lam * np.random.normal(loc=0, scale=sig, size=(neg_idxs.shape[0], ))
    return u


def save_exp_data(fdir, fname, exp_dict):
    if fdir[-1] != "/":
        fdir += "/"
    if not fname.endswith(".h5"):
        fname += ".h5"
    hf = h5py.File("{0}{1}".format(fdir, fname), "w")
    for k, v in exp_dict.items():
        hf.create_dataset(k, data=v)
    hf.close()