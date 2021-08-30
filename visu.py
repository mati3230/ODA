import h5py
import numpy as np
from visu_utils import\
	render_pptk,\
	render_o3d
from sp_utils import\
	partition,\
	initial_partition,\
	unions_to_partition
from io_utils import load_colors
import open3d as o3d


def render_point_cloud(P, colors, partition_vec=None):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(P[:, :3])
	col_mat = np.zeros((P.shape[0], 3))
	superpoints = np.unique(partition_vec)
	n_superpoints = superpoints.shape[0]
	for i in range(n_superpoints):
		superpoint_value = superpoints[i]
		idx = np.where(partition_vec == superpoint_value)[0]
		color = colors[i, :]
		col_mat[idx, :] = color
	pcd.colors = o3d.utility.Vector3dVector(col_mat)
	render_o3d(pcd, w_co=False)


def main():
	colors = load_colors()
	colors = colors/255.

	hf = h5py.File("D:/Projekte/Marcel/Partition/s3dis/graphs/conferenceRoom_1.h5", "r")
	P = np.array(hf["P"], copy=True)
	P[:, 3:] *= 0.5
	P[:, 3:] += 0.5
	P[:, 3:] *= 255

	node_features = np.array(hf["node_features"], copy=True)
	senders = np.array(hf["senders"], copy=True)
	receivers = np.array(hf["receivers"], copy=True)
	unions = np.array(hf["unions"], copy=True)
	partition_vec = np.array(hf["partition_vec"], copy=True)
	n_sps = hf["n_sps"][0]
	#print(n_sps)
	sp_idxs = n_sps * [None]
	for i in range(n_sps):
		sp_i = np.array(hf[str(i)], copy=True)
		sp_idxs[i] = sp_i

	graph_dict = {
		"nodes": node_features,
		"senders": senders,
		"receivers": receivers,
		"edges": None,
		"globals": None
	}

	init_p = initial_partition(P=P, sp_idxs=sp_idxs)
	my_p_vec = partition(
		graph_dict=graph_dict,
		unions=unions,
		P=P,
		sp_idxs=sp_idxs)
	#my_p_vec = unions_to_partition(graph_dict=graph_dict, unions=unions, sp_idxs=sp_idxs, P=P)
	viewer = render_pptk(P=P, initial_partition=init_p, partition=my_p_vec, point_size=0.03, v=None, colors=colors)
	viewer.close()
	#render_point_cloud(P=P, colors=colors, partition_vec=partition_vec)

if __name__ == "__main__":
	main()