import argparse
from tqdm import tqdm
import os
from p_tqdm import p_map
import numpy as np

from io_utils import load_gt_partition


def work(wargs):
    scene_id = wargs[0]
    fpath = wargs[1]

    gt = load_gt_partition(fpath=fpath)
    uni_gt = np.unique(gt)
    return len(uni_gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_dir", default="./partitions_files", type=str, help="Directory where we save the partiton files.")
    parser.add_argument("--csv_dir", default="./csvs", type=str, help="Directory where we save the csv.")
    parser.add_argument("--csv_name", default="number_gtos.csv", type=str, help="filename of the csv.")
    parser.add_argument("--n_proc", default=1, type=int, help="Number of processes that will be used.")
    args = parser.parse_args()

    files = os.listdir(args.partition_dir)
    n_files = len(files)
    workload = n_files * [None]
    for i in range(n_files):
        file = files[i]
        fpath = args.partition_dir + "/" + file
        scene_id = int(file.split("_")[0])
        workload[i] = (scene_id, fpath)
    res = p_map(work, workload, num_cpus=args.n_proc)
    arr = np.array(res, dtype=np.uint32)
    np.savetxt(fname=args.csv_name, arr=arr)


if __name__ == "__main__":
    main()