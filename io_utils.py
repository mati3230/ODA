import numpy as np
import open3d as o3d
import math
import os


def check_color_value(c):
    if c > 255 or c < 0:
        raise Exception("Invalid color value: {0}".format(c))


def load(file, r=255, g=0, b=0, p=1):
    if not os.path.exists(file):
        raise Exception("File not found: {0}".format(file))
    check_color_value(r)
    check_color_value(g)
    check_color_value(b)
    if file.endswith("npz"):
        P, pcd = from_npz(file=file, r=r, g=g, b=b, p=p)
    elif file.endswith("pcd"):
        P, pcd = from_pcd(file=file, r=r, g=g, b=b, p=p)
    elif file.endswith("txt"):
        P, pcd = from_txt(file=file, r=r, g=g, b=b, p=p)
    else:
        raise Exception("Unknwon file type: {0}".format(file))
    return P, pcd


def cloud(P, pcd, r=255, g=0, b=0, p=1):
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    colors_available = P.shape[1] == 6
    P, pcd = colorize(P=P, pcd=pcd, colors_available=colors_available, r=r, g=g, b=b)
    P, pcd = subsample(P=P, p=p, pcd=pcd)
    return P, pcd


def from_txt(file, r=255, g=0, b=0, p=1, delimiter=" "):
    pcd = o3d.geometry.PointCloud()
    P = np.loadtxt(file, delimiter=delimiter)
    P, pcd = cloud(P=P, pcd=pcd, r=r, g=g, b=b, p=p)
    return P, pcd


def from_npz(file, r=255, g=0, b=0, p=1):
    pcd = o3d.geometry.PointCloud()
    data = np.load(file, allow_pickle=True)
    P = data["P"]
    P, pcd = cloud(P=P, pcd=pcd, r=r, g=g, b=b, p=p)
    return P, pcd


def from_pcd(file, r=255, g=0, b=0, p=1):
    pcd = o3d.io.read_point_cloud(file)
    P = np.asarray(pcd.points)
    if pcd.colors is not None:
        C = pcd.colors
        P = np.hstack((P, C))
    P, pcd = cloud(P=P, pcd=pcd, r=r, g=g, b=b, p=p)
    return P, pcd


def colorize(P, pcd, colors_available, r=255, g=0, b=0):
    if not colors_available:
        C = np.zeros((P.shape[0], 3))
        C[:,0] = r
        C[:,1] = g
        C[:,2] = b
        pcd.colors = o3d.utility.Vector3dVector(C)
        P = np.hstack((P, C))
    return P, pcd


def subsample(P, p, pcd):
    if p > 1 or p <= 0:
        raise Exception("Cannot sample {0}% of the data.".format(p))
    if p == 1:
        return P, pcd
    size = math.floor(p * P.shape[0])
    idxs = np.arange(P.shape[0])
    idxs = np.random.choice(a=idxs, size=size, replace=False)
    P = P[idxs]
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(P[:, 3:])
    return P, pcd
