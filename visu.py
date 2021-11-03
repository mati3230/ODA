import h5py
import numpy as np
from visu_utils import\
	render_o3d
from io_utils import load_proc_cloud, load_partition, load_colors
import open3d as o3d
import argparse


def render_point_cloud(
        P, colors, partition_vec=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    if partition_vec is not None:
        col_mat = np.zeros((P.shape[0], 3))
        superpoints = np.unique(partition_vec)
        n_superpoints = superpoints.shape[0]
    
        for i in range(n_superpoints):
            superpoint_value = superpoints[i]
            idx = np.where(partition_vec == superpoint_value)[0]
            color = colors[i, :]
            col_mat[idx, :] = color
        pcd.colors = o3d.utility.Vector3dVector(col_mat)
    else:
        try:
            # print(P[:5, 3:6] / 255.0)
            pcd.colors = o3d.utility.Vector3dVector(P[:, 3:6] / 255.0)
        except Exception as e:
            print(e)
    render_o3d(pcd, w_co=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--load_partition", default=False, type=bool, help="")
    args = parser.parse_args()
    colors = load_colors()
    colors = colors/255.

    P = load_proc_cloud(fdir=args.g_dir, fname=args.g_filename)
    partition_vec = None
    if args.load_partition:
    	partition_vec = load_partition(file="{0}/partition_{1}.h5".format(args.g_dir, args.g_filename))
    render_point_cloud(P=P, colors=colors, partition_vec=partition_vec)

if __name__ == "__main__":
    main()