import os
import numpy as np
import argparse
from scipy.sparse.linalg import eigsh
from os import path
from pathlib import Path
from ogb.nodeproppred import DglNodePropPredDataset
from dataloader import(
    load_npz_to_sparse_graph,
    SparseGraph, 

)

from data_preprocess import (
    normalize_adj,
    eliminate_self_loops_adj,
    largest_connected_components,
    binarize_labels,
    laplacian_matrix
)

CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]

def write_cpf_engvectors(dataset, dataset_path):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.npz")
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")
    #?
    # remove self loop 
    data = data.standardize()
    
    adj,features,labels = data.unpack()
    lap = laplacian_matrix(adj)
    k = int(0.02 * adj.shape[0])
    print(k)
    #_, SM_eigenvectors = eigsh(lap, k = 100, which = 'SM' )
    #_, LM_eigenvectors = eigsh(lap, k = 100, which = 'LM')
    _, SM_eigenvectors = eigsh(lap, k, which = 'SM' )
    _, LM_eigenvectors = eigsh(lap, k, which = 'LM')
    eigenvectors = np.concatenate((LM_eigenvectors,SM_eigenvectors),axis=1 )
    out_path = Path.cwd().joinpath(dataset_path, f"{dataset}_egvs")
    
    with open(out_path, mode='w') as file_obj:
        for eigenvector in eigenvectors:
            for i in range(len(eigenvector)):
                if i != len(eigenvector)-1:
                    file_obj.write(str(eigenvector[i])+',')
                else:
                    file_obj.write(str(eigenvector[i]))
            file_obj.write('\n')
    return eigenvectors, features
    
def write_ogb_engvectors(dataset, dataset_path):
    data = DglNodePropPredDataset(dataset, dataset_path)
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g, labels = data[0]
    adj = g.adjacency_matrix(scipy_fmt='csr')
    k = int(0.02 * adj.shape[0])
    print(k)
    lap = laplacian_matrix(adj)
    _, SM_eigenvectors = eigsh(lap, k, which = 'SM' )
    _, LM_eigenvectors = eigsh(lap, k, which = 'LM')
    eigenvectors = np.concatenate((LM_eigenvectors,SM_eigenvectors),axis=1 )
    out_path = Path.cwd().joinpath(dataset_path, f"{dataset}_egvs")
    
    with open(out_path, mode='w') as file_obj:
        for eigenvector in eigenvectors:
            for i in range(len(eigenvector)):
                if i != len(eigenvector)-1:
                    file_obj.write(str(eigenvector[i])+',')
                else:
                    file_obj.write(str(eigenvector[i]))
            file_obj.write('\n')
    return eigenvectors, features

def write_engvectors(dataset, dataset_path):
    if dataset in CPF_data:
        return write_cpf_engvectors(dataset, dataset_path)
    elif dataset in OGB_data:
        return write_ogb_engvectors(dataset, dataset_path)

def get_args():
    parser = argparse.ArgumentParser(description="Write graph eigenvectors in file")
    parser.add_argument("--dataset_path", type=str, default='', help="data set path")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset_path = args.dataset_path
    datasets = ["a-computer", "a-photo"]
    #datasets = ["citeseer","pubmed","a-computer","a-photo", "ogbn-arxiv", "ogbn-products"]
    for dataset in datasets:
        write_engvectors(dataset, dataset_path)

if __name__ == "__main__":
    main()

#dataset_path = 'C:\\Users\\azhi\\Desktop\\hw\\Graduation_project\\myCode\\data'   

