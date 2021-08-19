import argparse

from io_utils import load_probs, load_unions, save_meshes, load_cloud
from sp_utils import get_objects, get_remaining, reconstruct, partition, initial_partition, reco, to_reco_params
from visu_utils import pick_sp_points_pptk, render_pptk, render_o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--o_dir", default="./out", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--save_meshes", default=False, type=bool, help="Should the meshes be stored in the o_dir?")
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--point_size", default=0.03, type=float, help="Rendering point size.")
    args = parser.parse_args()
    #"""
    point_size=args.point_size

    P, graph_dict, sp_idxs, probs, unions = load_probs(fdir=args.g_dir, filename=args.g_filename)
    unions, graph_dict = load_unions(fdir=args.g_dir, graph_dict=graph_dict, filename=args.g_filename)

    init_p = initial_partition(P=P, sp_idxs=sp_idxs)
    part = partition(
        graph_dict=graph_dict,
        unions=unions,
        P=P,
        sp_idxs=sp_idxs)
    picked_points_idxs, viewer = pick_sp_points_pptk(P=P, initial_partition=init_p, partition=part, point_size=point_size)
    viewer.close()
    # list of point idxs
    objects = get_objects(picked_points_idxs=picked_points_idxs, P=P, sp_idxs=sp_idxs, graph_dict=graph_dict, unions=unions)
    remaining = get_remaining(P=P, objects=objects)
    #"""
    # TODO del
    #P = load_cloud(file="../conferenceRoom_1.txt", r=255, g=0, b=0, p=0.025)
    #objects = []
    objects.append(remaining)
    for i in range(len(objects)):
        o_idxs = objects[i]
        while True:
            inp = input("Continue [c] | Parameter [dims(int),radius(float),sample_size(int)]: ")
            if inp == "c":
                break
            ok, dims, radius, sample_size = to_reco_params(inp=inp)
            if not ok:
                continue
            file = "{0}/mesh_{1}_{2}.obj".format(args.o_dir, args.g_filename, i)
            reco(P=P, o_idxs=o_idxs, file=file, dims=dims, radius=radius, sample_size=sample_size)



if __name__ == "__main__":
    main()