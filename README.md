# Requirements

* Python 3.6.8

# Install

* pip install -r requirements.txt

# Quickstart

python oda.py --file ../conferenceRoom_1.txt --p 0.05 --reg_strength 0.03 --initial_db 0.9 --load_probs True --g_filename a1cr1
python oda.py --file D:/Datasets/HSD/data/Hive/hive.npz --reg_strength 0.3 --initial_db 0.88 --g_filename hive --max_sp_size 25000 --p 0.1 --save_init_g True --save_probs True 

TODO: Add script to create the colors.npz file

## Windows

Ensure that the following files are in the project: 
* boost_numpy36-vc142-mt-x64-1_74.dll
* boost_python36-vc142-mt-x64-1_74.dll
* ply_c_ext.pyd
* cp_ext.pyd

# Code Insights of the oda.py Script

## Loading of a Point Cloud

At the beginning of the code a point cloud is loaded. This is realized with the helper function load_cloud which is defined in the [io_utils](./io_utils.py) script. After that a point cloud can be processed (e.g. aligning to the coordinate axis or translate the point cloud in the origin) so that our assumptions to an input point cloud are fitted. 

## Creation of the Superpoint Graph

The superpoint graph is created by a function which is called graph (see [ai_utils.py](./ai_utils.py)). The superpoint graph which is later passed to the neural network is created by functions of [Landrieu et al.](https://github.com/loicland/superpoint_graph). The end of the function superpoint_graph (see [ai_utils.py](./ai_utils.py)) returns the used graph data structure. 