# Requirements

* Python 3.6.8

# Install

* pip install -r requirements.txt
* pip install git+git://github.com/deepmind/graph_nets.git

# Quickstart

python oda.py --file ../conferenceRoom_1.txt --p 0.05 --reg_strength 0.03 --initial_db 0.9 --load_probs True --g_filename a1cr1
python oda.py --file D:/Datasets/HSD/data/Hive/hive.npz --reg_strength 0.3 --initial_db 0.87 --g_filename hive --max_sp_size 25000 --p 0.1