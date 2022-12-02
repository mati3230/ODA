# Graph Neural Network Partition

This repository contains a script [oda.py](./oda.py) to interactively create a partition. Furthermore, it contains scripts to evaluate graph neural network architectures and union algorithms in the context of the partition task. The strongly connected components algorithm and the segmentation algorithm of Felzenzwalb and Huttenlocher can be used as union algorithms. The code works on Windows. 

## Requirements

* Python 3.6.8

# Install

```
conda install -c anaconda boost
pip install -r requirements.txt
```

## Quickstart Partition Creation

```
python oda.py --file path/to/txt/point/cloud.txt --p 0.05 --reg_strength 0.03 --initial_db 0.9 --load_probs True --g_filename a1cr1
```
```
python oda.py --file path/to/npz/point/cloud.npz --reg_strength 0.3 --initial_db 0.88 --g_filename hive --max_sp_size 25000 --p 0.1 --save_init_g True --save_probs True 
```

## Evaluation

The evaluation can be conducted with the ScanNet dataset. After downloading the ScanNet dataset, create an environment variable SCANNET_DIR which points to the root directory of the dataset. For instance, SCANNET_DIR/scans should be a valid path.

First, create a preprocessed dataset to accelerate the evaluation process: 
```
python create_exp_dataset.py
```

After that, you can run an evaluation script, such as
```
python cc_vs_fh04.py
```
which executes that connected components and the segmentation algorithm of Felzenzwalb and Huttenlocher. The evaluation data will be stored in a folder called *csvs_cc_vs_fh04*. The csvs can be concatenated with:
```
python concat_csvs.py --csv_dir ./csvs_cc_vs_fh04 --file cc_vs_fh04.csv
```
After that, you can start the *cc_vs_fh04.ipynb* jupyter notebook to analyize the data. 

The other evaluation scripts ([correl_vs_gnn.py](./correl_vs_gnn.py) and [fh04_vary_k.py](./fh04_vary_k.py)) can be used in the same way.
