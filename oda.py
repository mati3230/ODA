import numpy as np
import open3d as o3d
import argparse
import os

from visu_utils import visualizer
from io_utils import load
from ai_utils import superpoint_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="File where the objects should be seperated.")
    parser.add_argument("--gpu", type=bool, default=False, help="Should the AI use the GPU. ")
    parser.add_argument("--r", type=int, default=255, help="Red channel if no color is available.")
    parser.add_argument("--g", type=int, default=0, help="Green channel if no color is available.")
    parser.add_argument("--b", type=int, default=0, help="Blue channel if no color is available.")
    parser.add_argument("--p", type=float, default=1, help="Subsampling factor.")
    args = parser.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    P, pcd = load(file=args.file, r=args.r, g=args.g, b=args.b, p=args.p)
    print(P.shape)
    """
    vis = visualizer()
    vis.run()
    vis.destroy_window()
    """
    superpoint_graph(xyz=P[:, :3], rgb=P[:, 3:])


if __name__ == "__main__":
    main()