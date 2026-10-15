import os
import torch
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ipdb
import json
from scipy.special import softmax
import torch.nn as nn
import torch.nn.functional as F
import faiss
from kmeans_pytorch import kmeans
import pandas as pd
# from spherecluster import SphericalKMeans
# from sklearn_extra.cluster import KMeans as SphericalKMeans

def load_tasks(args):
    with open(os.path.join(args.base_dir, 'data', 'file1_665.json')) as f:
        file1_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file2_665.json')) as f:
        file2_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file3_665.json')) as f:
        file3_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file4_665.json')) as f:
        file4_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file5_665.json')) as f:
        file5_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file6_665.json')) as f:
        file6_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file7_665.json')) as f:
        file7_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file8_665.json')) as f:
        file8_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file9_665.json')) as f:
        file9_665 = json.load(f)
    with open(os.path.join(args.base_dir, 'data', 'file10_665.json')) as f:
        file10_665 = json.load(f)
    return file1_665, file2_665, file3_665, file4_665, file5_665, file6_665, file7_665, file8_665, file9_665, file10_665

def get_nn(features, num_neighbors):
    """
    Calculates nearest neighbors using FAISS with cosine similarity.

    Args:
        features (ndarray): L2-normalized feature vectors.
        num_neighbors (int): Number of nearest neighbors to find.

    Returns:
        similarities (ndarray): Cosine similarities to nearest neighbors.
        indices (ndarray): Indices of nearest neighbors.
    """
    d = features.shape[1]
    features = features.astype(np.float32)
    index = faiss.IndexFlatIP(d)  # Use inner product index
    index.add(features)  # Add vectors to the index
    similarities, indices = index.search(features, num_neighbors + 1)
    # Remove self-matching
    similarities = similarities[:, 1:]
    indices = indices[:, 1:]
    return similarities, indices

def get_mean_nn_similarity(features, num_neighbors):
    similarities, _ = get_nn(features, num_neighbors)
    mean_similarity = similarities.mean(axis=1)
    return mean_similarity

def calculate_typicality(features, num_neighbors):
    mean_similarity = get_mean_nn_similarity(features, num_neighbors)
    # Higher mean similarity indicates higher typicality
    typicality = mean_similarity
    return typicality

def perform_clustering(args, feature_vectors, n_clusters, clustering_output_path, clustering_method='kmeans'):
    """
    Args:
        feature_vectors (ndarray): Array of feature vectors.
        n_clusters (int): Number of clusters.
        clustering_output_path (str): Path to save clustering results.

    Returns:
        labels (ndarray): Cluster labels for each feature vector.
        kmeans : Trained Kmeans model.
    """
    print('Performing KMeans clustering...')
    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=12345)
        labels = kmeans.fit_predict(feature_vectors)
    # elif clustering_method == 'spherical_kmeans':
    #     kmeans = SphericalKMeans(n_clusters=n_clusters, random_state=12345)
    #     labels = kmeans.fit_predict(feature_vectors)
        

    # Save clustering results
    os.makedirs(clustering_output_path, exist_ok=True)
    np.save(os.path.join(clustering_output_path, f'cluster_labels_file{args.task_num}.npy'), labels)
    np.save(os.path.join(clustering_output_path, f'cluster_centers_file{args.task_num}.npy'), kmeans.cluster_centers_)
    print(f'Clustering results saved to {clustering_output_path}')

    return labels, kmeans

# def perform_clustering(args, feature_vectors, n_clusters, clustering_output_path, clustering_method='kmeans'):
    
#     print('Performing KMeans clustering with kmeans-pytorch...')
#     if clustering_method == 'kmeans':
#         # Convert feature_vectors to a PyTorch tensor and move to GPU
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         data = torch.from_numpy(feature_vectors).float().to(device)

#         # Perform k-means clustering
#         cluster_ids_x, cluster_centers = kmeans(
#             X=data, num_clusters=n_clusters, distance='euclidean', device=device
#         )

#         # Move results back to CPU and convert to numpy arrays
#         labels = cluster_ids_x.cpu().numpy()
#         cluster_centers = cluster_centers.cpu().numpy()

#     # Save clustering results
#     os.makedirs(clustering_output_path, exist_ok=True)
#     np.save(os.path.join(clustering_output_path, f'cluster_labels_file{args.task_num}.npy'), labels)
#     np.save(os.path.join(clustering_output_path, f'cluster_centers_file{args.task_num}.npy'), cluster_centers)
#     print(f'Clustering results saved to {clustering_output_path}')

#     return labels, cluster_centers

def load_clustering_results(args, clustering_output_path):
    """
    Load clustering results from files.

    Args:
        clustering_output_path (str): Path where clustering results are saved.

    Returns:
        labels (ndarray): Cluster labels for each feature vector.
        cluster_centers (ndarray): Cluster centers.
    """
    labels_path = os.path.join(clustering_output_path, f'cluster_labels_file{args.task_num}.npy')
    centers_path = os.path.join(clustering_output_path, f'cluster_centers_file{args.task_num}.npy')

    if os.path.exists(labels_path) and os.path.exists(centers_path):
        labels = np.load(labels_path)
        cluster_centers = np.load(centers_path)
        print(f'Loaded clustering results from {clustering_output_path}')
        return labels, cluster_centers
    else:
        print('Clustering results not found.')
        return None, None
    
    

def allocate_samples(task_difficulties, task_dict, total_budget, T):
    """
    Allocates samples to tasks based on their difficulties and available samples,
    ensuring that the total allocations match the total_budget.

    Args:
        task_difficulties (dict): A dictionary mapping task names to their difficulty values.
        task_dict (dict): A dictionary mapping task names to their list of samples.
        total_budget (int): The total number of samples to allocate across all tasks.
        T (float): Temperature parameter for softmax scaling.

    Returns:
        adjusted_allocations (dict): A dictionary mapping task names to the number of allocated samples.
    """

    # Total samples available per task from task_dict
    task_total_samples = {task_name: len(samples) for task_name, samples in task_dict.items()}

    # Calculate total available samples
    total_available_samples = sum(task_total_samples.values())
    if total_available_samples < total_budget:
        print(f"Total available samples ({total_available_samples}) are less than the total budget ({total_budget}). Adjusting total_budget to {total_available_samples}.")
        total_budget = total_available_samples

    # Extract task difficulties and names
    task_names = list(task_difficulties.keys())
    task_difficulties_np = np.array([task_difficulties[task] for task in task_names])

    # Compute softmax weights
    softmax_weights = F.softmax(-torch.tensor(task_difficulties_np) / T, dim=0).numpy()

    # Initial allocation based on weights (exact allocations)
    initial_allocations = {task: weight * total_budget for task, weight in zip(task_names, softmax_weights)}

    # Take the floor of each allocation
    floor_allocations = {task: int(np.floor(count)) for task, count in initial_allocations.items()}

    # Calculate the surplus
    total_floor_alloc = sum(floor_allocations.values())
    surplus = total_budget - total_floor_alloc

    # Compute fractional parts
    fractional_parts = {task: initial_allocations[task] - floor_allocations[task] for task in task_names}

    # Sort tasks based on fractional parts in descending order
    sorted_tasks = sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)

    # Initialize adjusted allocations with floor allocations
    adjusted_allocations = floor_allocations.copy()

    # Distribute surplus
    for task, fraction in sorted_tasks:
        if surplus <= 0:
            break
        max_additional = task_total_samples[task] - adjusted_allocations[task]
        if max_additional > 0:
            adjusted_allocations[task] += 1
            surplus -= 1

    # Adjust allocations based on available samples
    for task in task_names:
        available = task_total_samples.get(task, 0)
        if adjusted_allocations[task] > available:
            surplus += adjusted_allocations[task] - available
            adjusted_allocations[task] = available

    # If there's surplus after adjustments, redistribute if possible
    if surplus > 0:
        remaining_tasks = [task for task in task_names if adjusted_allocations[task] < task_total_samples[task]]
        while surplus > 0 and remaining_tasks:
            for task in remaining_tasks:
                if surplus <= 0:
                    break
                max_additional = task_total_samples[task] - adjusted_allocations[task]
                if max_additional > 0:
                    adjusted_allocations[task] += 1
                    surplus -= 1
            # Update remaining_tasks
            remaining_tasks = [task for task in task_names if adjusted_allocations[task] < task_total_samples[task]]
    # Final total allocated samples
    total_allocated = sum(adjusted_allocations.values())
    print(f"Total Allocated Samples: {total_allocated}")

    return adjusted_allocations