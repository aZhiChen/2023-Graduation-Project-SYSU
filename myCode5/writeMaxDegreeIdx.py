import os
import numpy as np
import argparse
from scipy.sparse.linalg import eigsh
from os import path
from pathlib import Path

from dataloader import(
    load_npz_to_sparse_graph,
    SparseGraph, 

)

from data_preprocess import (
    normalize_adj,
    eliminate_self_loops_adj,
    largest_connected_components,
    binarize_labels,
    laplacian_matrix,
    max_idx,
)


def Write_MaxDegreeIdx(dataset, dataset_path):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.npz")
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")
    #?
    # remove self loop 
    data = data.standardize()
    
    adj,features,labels = data.unpack()
    #lap = laplacian_matrix(adj)
    print(type(adj))
    A = np.array(adj.todense())
    d = np.sum(A, axis=1)
    '''
    for i in range(len(degree)):
        if degree[i] != 0:
            print(degree[i])
    '''
    degree = d.tolist() 
    idx_list = max_idx(degree, len(degree))
    out_path = Path.cwd().joinpath(dataset_path, f"{dataset}_MaxDegreeIdx")
    
    with open(out_path, mode='w') as file_obj:
        for i in range(len(idx_list)):
            if i != len(idx_list)-1:
                file_obj.write(str(idx_list[i])+',')
            else:
                file_obj.write(str(idx_list[i]))
            
    return idx_list

def get_args():
    parser = argparse.ArgumentParser(description="Write graph eigenvectors in file")
    parser.add_argument("--dataset_path", type=str, default='C:\\Users\\azhi\\Desktop\\hw\\Graduation_project\\myCode3\\data\\', help="data set path")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset_path = args.dataset_path
    #datasets = ["cora"]
    datasets = ["cora","citeseer","pubmed","a-computer","a-photo"]
    for dataset in datasets:
        Write_MaxDegreeIdx(dataset, dataset_path)

if __name__ == "__main__":
    main()

#dataset_path = 'C:\\Users\\azhi\\Desktop\\hw\\Graduation_project\\myCode\\data'   

