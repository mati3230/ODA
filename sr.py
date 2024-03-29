import argparse

from io_utils import load_init_graph, load_unions, save_mesh, load_cloud, load_colors, save_ply
from sp_utils import get_objects, get_remaining, partition, initial_partition, reco, to_reco_params, simplify_mesh, recenter
from visu_utils import pick_sp_points_pptk, render_pptk, render_o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_dir", default="./tmp", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--o_dir", default="./out", type=str, help="Directory where the graphs will be stored.")
    parser.add_argument("--save_meshes", default=False, type=bool, help="Should the meshes be stored in the o_dir?")
    parser.add_argument("--g_filename", default="", type=str, help="Filename will be used as a postfix.")
    parser.add_argument("--point_size", default=0.03, type=float, help="Rendering point size.")
    parser.add_argument("--ending", default="glb", type=str, help="File format of the meshes (glb, obj, ...)")
    parser.add_argument("--co", default=False, type=bool, help="Visualize an o3d coordinate system")
    parser.add_argument("--ply", default=False, type=bool, help="Only save ply files")
    #parser.add_argument("--all", default=False, type=bool, help="Triangulate all meshes")
    args = parser.parse_args()
    #"""
    point_size=args.point_size
    colors = load_colors()
    colors = colors/255.

    P, graph_dict, sp_idxs = load_init_graph(fdir=args.g_dir, filename=args.g_filename, half="")
    unions, graph_dict = load_unions(fdir=args.g_dir, graph_dict=graph_dict, filename=args.g_filename)

    print("Center point cloud.")
    P = recenter(P=P)

    init_p = initial_partition(P=P, sp_idxs=sp_idxs)
    part = partition(
        graph_dict=graph_dict,
        unions=unions,
        P=P,
        sp_idxs=sp_idxs)
    picked_points_idxs, viewer = pick_sp_points_pptk(
        P=P, initial_partition=init_p, partition=part, point_size=point_size, colors=colors)
    #save_picked_points(picked_points_idxs)
    viewer.close()
    # list of point idxs
    objects = get_objects(picked_points_idxs=picked_points_idxs, P=P, sp_idxs=sp_idxs, graph_dict=graph_dict, unions=unions)
    remaining = get_remaining(P=P, objects=objects)
    objects.append(remaining)
    meshes = []
    mesh = None
    for i in range(len(objects)):
        print("Object: {0}/{1}".format(i+1, len(objects)))
        last_one = i == len(objects) - 1
        o_idxs = objects[i]
        if args.ply:
            O = P[o_idxs]
            save_ply(P=O, fdir=args.o_dir, filename=args.g_filename, o_id=i)
        else:
            while True:
                #"""
                inp = input("Continue [c] | Simplify [s] | Parameter [dims(int),radius(float),sample_size(int)]: ")
                if inp == "s":
                    if mesh is None:
                        continue
                    mesh = simplify_mesh(mesh)
                elif inp == "c":
                    break
                else:
                    ok, dims, radius, sample_size = to_reco_params(inp=inp)
                    if not ok:
                        continue
                    """            
                    dims=100 
                    radius=0.1
                    sample_size=10
                    """
                    mesh = reco(P=P, o_idxs=o_idxs, dims=dims, radius=radius, sample_size=sample_size)
                if mesh is None:
                    continue
                if args.save_meshes:
                    ending = args.ending
                    if last_one:
                        ending = "_room{0}".format(ending)
                    save_mesh(mesh=mesh, fdir=args.o_dir, filename=args.g_filename, o_id=i, ending=ending)
                render_o3d(mesh, w_co=args.co)
            meshes.append(mesh)
    if not args.ply:
        render_o3d(meshes, w_co=args.co)

if __name__ == "__main__":
    main()