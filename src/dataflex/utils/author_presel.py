# Vendored unmodified function bodies from the upstream MIT-licensed source.
# See upstream_licenses/ and docs/author_alignment.md for provenance.
import numpy as np
import faiss

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
