import argparse

from io_utils import load_probs, load_unions, load_colors, save_meshes
from sp_utils import get_objects, get_remaining, reconstruct
from visu_utils import partition_pcd, pick_sp_points, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--o_dir", default="./out", type=str, help="Directory where the graphs will be stored.")
    args = parser.parse_args()
    
    colors = load_colors()

    P, graph_dict, sp_idxs, probs, unions = load_probs(fdir=args.g_dir)
    unions, graph_dict = load_unions(fdir=args.g_dir, graph_dict=graph_dict)

    p_pcd = partition_pcd(graph_dict=graph_dict, unions=unions, P=P, sp_idxs=sp_idxs, colors=colors, col_d=-1)
    #picked_points_idxs = pick_sp_points(p_pcd)
    picked_points_idxs = [20378, 273, 11412, 36658, 3403, 20506, 15024, 9035, 45127]

    objects = get_objects(picked_points_idxs=picked_points_idxs, P=P, sp_idxs=sp_idxs, graph_dict=graph_dict, unions=unions)

    remaining = get_remaining(P=P, objects=objects)
    #render(P[remaining])
    meshes = reconstruct(P=P, objects=objects, remaining=remaining)
    render(meshes)
    save_meshes(meshes=meshes, fdir=args.o_dir)


if __name__ == "__main__":
    main()