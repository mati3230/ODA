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