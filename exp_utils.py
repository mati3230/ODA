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


def match_score(gt_partition, partition, ignore=None, weighted=True, limitation=4, return_ooa=False):
    classification, densities = gt_partition.classify(partition_B=partition, density_function=densities_np)

    # ignore matches in the pair classification
    if ignore is not None:
        for i in ignore:
            classification[classification == i] = 0

    # number of points
    max_density = gt_partition.partition.shape[0]

    n_matches = np.where(classification > 0)[0].shape[0]
    match_stats = 3*[0]
    dens_stats = 3*[0]
    match_values = np.arange(4, dtype=np.int32)
    #print(match_values)
    for i in range(match_values.shape[0]):
        mv = match_values[i]
        if mv == 0:
            continue
        mv_idxs = np.where(classification == mv)
        n_mv = mv_idxs[0].shape[0]
        den_sum = np.sum(densities[mv_idxs[0], mv_idxs[1]])
        dens_stats[i-1] = "{0:.3f}".format(100*den_sum/max_density)
        match_stats[i-1] = n_mv

    # determine the value of a first order match
    foc = gt_partition.first_order_class
    if not weighted:
        classification[classification > 0] = 1
        foc = 1

    # calculate the reward
    weighted_sum = np.sum(densities * classification)
    max_sum = np.sum(foc * gt_partition.counts)
    raw_score = weighted_sum / max_sum

    # determine the limitation
    L = max(classification.shape[0] / classification.shape[1], classification.shape[1] / classification.shape[0])**(limitation)
    score = raw_score / L
    if return_ooa:
        ooa, _ = gt_partition.overall_obj_acc(max_density=max_density, partition_B=partition, densities=densities)
        return score, raw_score, ooa, match_stats, dens_stats
    return score, raw_score, match_stats, dens_stats


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
    y_ = np.array(y, copy=True)
    y_ = y_.astype(np.float32)
    zero_idxs = np.where(probs == 0)[0]
    probs[zero_idxs] = eps
    pos_idxs = np.where(y_ == 1)[0]
    pos_bce = -np.log(probs[pos_idxs])

    zero_idxs = np.where((1-probs) == 0)[0]
    probs[zero_idxs] = 1-eps
    neg_idxs = np.where(y_ == 0)[0]
    neg_bce = -np.log(1-probs[neg_idxs])

    bce = np.vstack((pos_bce[:, None], neg_bce[:, None]))
    bce = np.abs(np.mean(bce))
    #print(bce)
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


def load_exp_data(fdir, fname):
    if fdir[-1] != "/":
        fdir += "/"
    if not fname.endswith(".h5"):
        fname += ".h5"
    hf = h5py.File("{0}{1}".format(fdir, fname), "r")
    exp_dict = {
        "node_features": np.array(hf["node_features"], copy=True),
        "senders": np.array(hf["senders"], copy=True),
        "receivers": np.array(hf["receivers"], copy=True),
        "probs_gnn": np.array(hf["probs_gnn"], copy=True),
        "probs_correl": np.array(hf["probs_correl"], copy=True),
        "probs_random": np.array(hf["probs_random"], copy=True),
        "probs_imperfect": np.array(hf["probs_imperfect"], copy=True),
        "p_gt": np.array(hf["p_gt"], copy=True),
        "p_cp": np.array(hf["p_cp"], copy=True),
        "unions_gt": np.array(hf["unions_gt"], copy=True),
        "bce_gnn": np.array(hf["bce_gnn"], copy=True),
        "bce_correl": np.array(hf["bce_correl"], copy=True),
        "bce_random": np.array(hf["bce_random"], copy=True),
        "bce_imperfect": np.array(hf["bce_imperfect"], copy=True)
    }
    sp_idxs = []
    size = hf["|S|"][0]
    for i in range(size):
        sp = np.array(hf[str(i)], copy=True)
        sp_idxs.append(sp)
    hf.close()
    return exp_dict, sp_idxs