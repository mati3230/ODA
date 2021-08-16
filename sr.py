import argparse

from io_utils import load_probs, load_unions, save_meshes
from sp_utils import get_objects, get_remaining, reconstruct, partition, initial_partition
from visu_utils import pick_sp_points_pptk, render_pptk, render_o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--o_dir", default="./out", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--save_meshes", default=False, type=bool, help="Should the meshes be stored in the o_dir?")
    parser.add_argument("--depth", default=13, type=int, help="Octree depth of the poisson reconstruction algorithm.")
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--point_size", default=0.03, type=float, help="Rendering point size.")
    args = parser.parse_args()

    point_size=args.point_size

    P, graph_dict, sp_idxs, probs, unions = load_probs(fdir=args.g_dir, filename=args.g_filename)
    unions, graph_dict = load_unions(fdir=args.g_dir, graph_dict=graph_dict, filename=args.g_filename)

    init_p = initial_partition(P=P, sp_idxs=sp_idxs)
    part = partition(
        graph_dict=graph_dict,
        unions=unions,
        P=P,
        sp_idxs=sp_idxs)
    picked_points_idxs = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size)

    objects = get_objects(picked_points_idxs=picked_points_idxs, P=P, sp_idxs=sp_idxs, graph_dict=graph_dict, unions=unions)

    remaining = get_remaining(P=P, objects=objects)
    #render(P[remaining])
    meshes = reconstruct(P=P, objects=objects, remaining=remaining, depth=args.depth)
    render_o3d(meshes)
    if args.save_meshes:
        save_meshes(meshes=meshes, fdir=args.o_dir, filename=args.g_filename)


if __name__ == "__main__":
    main()