import os
import torch
import sys
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import ipdb
import json
from scipy.special import softmax
import torch.nn as nn
import torch.nn.functional as F
import faiss
sys.path.append("./data_process")
from clust_utils import *


def load_features(feature_files):
    """
    Load features from the given list of feature files.

    Args:
        feature_files (list of str): List of paths to feature files.

    Returns:
        unique_idxs (list): List of unique_idx identifiers.
        feature_vectors (ndarray): Array of feature vectors.
    """
    features = {}
    for file in feature_files:
        print(f'Loading features from {file}')
        data = torch.load(file)
        features.update(data)
    print(f'Total features loaded: {len(features)}')

    unique_idxs = list(features.keys())
    feature_vectors = [features[idx] for idx in unique_idxs]

    feature_vectors = np.stack(feature_vectors)
    return unique_idxs, feature_vectors

def select_samples(unique_idxs, labels, N, already_selected_idxs):
    """
    Select N samples proportionally from clusters, avoiding already selected samples.

    Args:
        unique_idxs (list): List of unique_idx identifiers.
        labels (ndarray): Cluster labels for each feature vector.
        N (int): Total number of samples to select.
        already_selected_idxs (set): Set of unique_idxs that are already selected.

    Returns:
        selected_unique_idxs (list): List of selected unique_idx identifiers.
    """
    unique_idxs = [int(t.item()) for t in unique_idxs]
    total_samples = len(unique_idxs)
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # Prepare data structures
    idx_to_unique_idx = {i: uid for i, uid in enumerate(unique_idxs)}
    unique_idx_to_idx = {uid: i for i, uid in enumerate(unique_idxs)}

    # Exclude already selected samples
    available_indices = set(range(total_samples)) - {unique_idx_to_idx[uid] for uid in already_selected_idxs if uid in unique_idx_to_idx}

    selected_indices = []

    # Calculate number of samples to select from each cluster
    total_available_samples = len(available_indices)
    cluster_sample_counts = {}

    for label in unique_labels:
        cluster_available_indices = set(cluster_indices[label]) & available_indices
        cluster_size = len(cluster_available_indices)
        proportion = cluster_size / total_available_samples if total_available_samples > 0 else 0
        samples_to_select = int(round(proportion * N))
        cluster_sample_counts[label] = min(samples_to_select, cluster_size)

    total_selected = sum(cluster_sample_counts.values())
    if total_selected < N:
        deficit = N - total_selected
        
        sorted_labels = sorted(unique_labels, key=lambda l: counts[l], reverse=True)
        for label in sorted_labels:
            cluster_available_indices = set(cluster_indices[label]) & available_indices
            cluster_size = len(cluster_available_indices)
            if cluster_sample_counts[label] < cluster_size:
                cluster_sample_counts[label] += 1
                deficit -=1
                if deficit == 0:
                    break
    elif total_selected > N:
        surplus = total_selected - N
        sorted_labels = sorted(unique_labels, key=lambda l: counts[l])
        for label in sorted_labels:
            if cluster_sample_counts[label] > 0:
                cluster_sample_counts[label] -= 1
                surplus -=1
                if surplus == 0:
                    break

    # select samples from each cluster
    for label in unique_labels:
        cluster_available_indices = list(set(cluster_indices[label]) & available_indices)
        num_to_select = cluster_sample_counts[label]
        if num_to_select > 0 and len(cluster_available_indices) > 0:
            selected_in_cluster = np.random.choice(cluster_available_indices, size=num_to_select, replace=False)
            selected_indices.extend(selected_in_cluster)

    selected_unique_idxs = [idx_to_unique_idx[i] for i in selected_indices]

    return selected_unique_idxs

def typical_sample_selection(unique_idxs, feature_vectors, labels, N, already_selected_idxs):
    unique_idxs = [int(t.item()) for t in unique_idxs]
    total_samples = len(unique_idxs)
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # Prepare data structures
    idx_to_unique_idx = {i: uid for i, uid in enumerate(unique_idxs)}
    unique_idx_to_idx = {uid: i for i, uid in enumerate(unique_idxs)}
    # Exclude already selected samples
    available_indices = set(range(total_samples)) - {unique_idx_to_idx[uid] for uid in already_selected_idxs if uid in unique_idx_to_idx}
    
    selected_indices = []

    # Calculate number of samples to select from each cluster
    total_available_samples = len(available_indices)
    cluster_sample_counts = {}
    for label in unique_labels:
        cluster_available_indices = set(cluster_indices[label]) & available_indices
        cluster_size = len(cluster_available_indices)
        proportion = cluster_size / total_available_samples if total_available_samples > 0 else 0
        samples_to_select = int(round(proportion * N))
        cluster_sample_counts[label] = min(samples_to_select, cluster_size)
        
    total_selected = sum(cluster_sample_counts.values())
    if total_selected < N:
        deficit = N - total_selected
        sorted_labels = sorted(unique_labels, key=lambda l: counts[l], reverse=True)
        for label in sorted_labels:
            cluster_available_indices = set(cluster_indices[label]) & available_indices
            cluster_size = len(cluster_available_indices)
            if cluster_sample_counts[label] < cluster_size:
                cluster_sample_counts[label] += 1
                deficit -=1
                if deficit == 0:
                    break
    elif total_selected > N:
        surplus = total_selected - N
        sorted_labels = sorted(unique_labels, key=lambda l: counts[l])
        for label in sorted_labels:
            if cluster_sample_counts[label] > 0:
                cluster_sample_counts[label] -= 1
                surplus -=1
                if surplus == 0:
                    break

    # select samples from each cluster based on typicality
    for label in unique_labels:
        cluster_available_indices = list(set(cluster_indices[label]) & available_indices)
        num_to_select = cluster_sample_counts[label]
        if num_to_select > 0 and len(cluster_available_indices) > 0:
            cluster_features = feature_vectors[cluster_available_indices]
            # Compute typicality
            typicality = calculate_typicality(cluster_features, min(5, len(cluster_available_indices)-1))
            # Get indices of samples with highest typicality
            sorted_indices = np.argsort(-typicality)
            selected_in_cluster = [cluster_available_indices[i] for i in sorted_indices[:num_to_select]]
            selected_indices.extend(selected_in_cluster)

    selected_unique_idxs = [idx_to_unique_idx[i] for i in selected_indices]

    return selected_unique_idxs

def main(args):
    # Load features
    feature_files = []
    feature_files = [args.features_path]
    # feature_files = args.features_path
    unique_idxs, feature_vectors = load_features(feature_files)
    # Normalize features
    feature_vectors_normalized = normalize(feature_vectors, norm='l2')
    
    clustering_output_path = args.clustering_output_path

    # number of clusters (1% of total samples)
    n_clusters = max(1, int(0.01 * len(unique_idxs)))
    # n_clusters = max(1, int(0.05 * len(unique_idxs)))
    print(f'Number of clusters: {n_clusters}')

    # Load clustering results or perform clustering
    labels, cluster_centers = load_clustering_results(args, clustering_output_path)
    if labels is None or cluster_centers is None:
        labels, kmeans = perform_clustering(args, feature_vectors_normalized, n_clusters, clustering_output_path, 'kmeans')
    else:
        pass 
    
    # reference set
    data_path = os.path.join(args.base_dir, 'data', 'round1_665k_notext.json')
    with open(data_path, 'r') as file:
        selected_samples = json.load(file)
    
    already_selected_idxs = {sample['unique_idx'] for sample in selected_samples}
    print(f'Loaded {len(already_selected_idxs)} already selected unique_idxs from {args.already_selected_file}')
    # Select N samples
    N = args.N
    if args.method == "random":
        selected_unique_idxs = select_samples(unique_idxs, labels, N, already_selected_idxs)
    elif args.method == "typical":
        selected_unique_idxs = typical_sample_selection(unique_idxs, feature_vectors_normalized, labels, N, already_selected_idxs)
    
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'new_samples_file{args.task_num}_rnd{args.round}_{args.method}_T_2sqrt_s12345.json')
    
    with open(save_path, 'w') as file:
        json.dump(selected_unique_idxs, file, indent=2)
    print(f"Saved {len(selected_unique_idxs)} samples to {save_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, default='',
                        help='Path to the features file (.pt) or directory containing feature files.')
    parser.add_argument('--task_num', type=str, default="10",
                        help='task number.')
    parser.add_argument('--clustering_output_path', type=str, default='/data/02/sylo/mnt_01/Projects/datasets/clustering_results/openclip-large/',
                        help='Path to save clustering results.')
    parser.add_argument('--save_path', type=str, default='/data/02/sylo/mnt_01/Projects/datasets/clustering_results/openclip-large/',
                        help='Path to save clustering results.')
    parser.add_argument('--method', type=str, default="random", choices=["random", "typical"], 
                        help="Method Name")
    parser.add_argument('--round', type=int, default=2,
                        help='round.')
    parser.add_argument('--base_dir', type=str, default='/data/01/sylo/',
                        help='base_dir')
    parser.add_argument('--N', type=int, default=5000,
                        help='Total number of samples to select.')
    parser.add_argument('--already_selected_file', type=str, default=None,
                        help='Path to file containing already selected unique_idxs (one per line).')
    parser.add_argument('--output_selected_file', type=str, default='selected_unique_idxs.txt',
                        help='Path to save the selected unique_idxs.')
    args = parser.parse_args()
    
    #modify args based on args.base_dir (main code for dino)
    args.clustering_output_path = os.path.join(args.base_dir, 'datasets', 'clustering_results', 'dino_5_perc_l2')
    args.save_path = os.path.join(args.base_dir, 'datasets', 'clustering_results', 'dino_5_perc_l2')
    
    data_path = os.path.join(args.base_dir, 'datasets', 'llava_v1_5_mix665k_notext_add_idx.json')
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    file1_665, file2_665, file3_665, file4_665, file5_665, file6_665, file7_665, file8_665, file9_665, file10_665 = load_tasks(args)

    # per round budget = 5% of total samples
    initial_num_samples = max(1, int(len(dataset) * 5 / 100))
    
    '''
    Size-balanced
    '''
    # size_dict = {} 
    # # Calculate and print for all files
    # for i, file in enumerate([file1_665, file2_665, file3_665, file4_665, file5_665, file6_665, file7_665, file8_665, file9_665, file10_665]):
    #     indices = [item['unique_idx'] for item in file]
    #     size_dict[f"file{i+1}"] = len(indices)
    # # Calculate the total number of samples across all files
    # total_size = sum(size_dict.values())
    
    # for i, file in enumerate([file1_665, file2_665, file3_665, file4_665, file5_665, file6_665, file7_665, file8_665, file9_665, file10_665]):
    #     args.task_num = f"{i + 1}"
    #     args.features_path = os.path.join(args.base_dir, 'mnt_01', 'Projects', 'datasets', 'features', f'dino_feat_chunk_0_file_{args.task_num}.pt')
         
    #     for rnd in [2,3,4]:
    #         args.round = rnd
    #         # Calculate the weight for each file based on its length
    #         weights_dict = {file: (size / total_size) * (rnd - 1) * initial_num_samples for file, size in size_dict.items()}
    #         weights = [int(weight) for weight in weights_dict.values()]
    #         args.N = weights[int(args.task_num) - 1]
    #         main(args)
    
    
    '''
    Loss-noinst-ratio-Softmax-negetive (PreSel)
    '''
    # Load task difficulties
    with open(os.path.join(args.base_dir, 'data', 'task_difficulties.json'), 'r') as f:
        task_difficulties = json.load(f)
    task_difficulties_np = np.array(task_difficulties)
    T = 1 / np.sqrt(10)
    for i, file in enumerate([file1_665, file2_665, file3_665, file4_665, file5_665, file6_665, file7_665, file8_665, file9_665, file10_665]):
        args.task_num = f"{i + 1}"
        args.features_path = os.path.join(args.base_dir, 'datasets', 'features', f'dino_feat_chunk_0_file_{args.task_num}.pt')
        for rnd in [2,3,4]: # Round 3 corresponds to 15% of total samples. You can change it to any other round.
            args.round = rnd
            N = (rnd - 1) * initial_num_samples

            softmax_diff = softmax(-task_difficulties_np/T)
            allocated_samples_hybrid = (softmax_diff * N).astype(int)
            difference_hybrid = N - allocated_samples_hybrid.sum()
            if difference_hybrid > 0:
                allocated_samples_hybrid[np.argmax(softmax_diff)] += difference_hybrid
            elif difference_hybrid < 0:
                allocated_samples_hybrid[np.argmin(softmax_diff)] += difference_hybrid
            args.N = allocated_samples_hybrid[int(args.task_num) - 1]
            print(f"Round {rnd}: {args.N}")
            print(f"Round {rnd}: {allocated_samples_hybrid}")
            main(args)
    
    
    
    
    

