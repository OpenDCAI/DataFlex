# Vendored unmodified function bodies from the upstream PreSel source.
# MIT License
#
# Copyright (c) 2025 Bardia Safaei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""PreSel's LLaVA record-ID selection, with no image-to-record expansion.

The author implementation operates on unique_idx even when image paths repeat.
Seed is configurable to retain the experiment's explicit seed=42 requirement.
Training settings are configured independently of the selection pipeline.
"""
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from sklearn.cluster import KMeans


def task_order(records):
    def key(task):
        suffix = str(task).rsplit('_', 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, str(task))
    return sorted({r['task'] for r in records}, key=key)


def reference_indices(records, seed=42):
    """Randomly sample ~5% of each task, totaling floor(0.05*N) records."""
    tasks = task_order(records)
    rows = [[i for i, r in enumerate(records) if r['task'] == task] for task in tasks]
    exact = np.asarray([len(x)*0.05 for x in rows])
    quotas = np.floor(exact).astype(int)
    budget = int(len(records)*0.05)
    # Per-task integer rounding only; never round image groups then expand.
    residual_order = np.argsort(-(exact-quotas), kind='stable')
    for i in residual_order[:budget-int(quotas.sum())]: quotas[i] += 1
    rng = random.Random(seed)
    selected = sorted(i for ids, q in zip(rows, quotas) for i in rng.sample(ids, int(q)))
    if len(selected) != budget or len(set(selected)) != budget:
        raise ValueError('Reference budget is not exact')
    return selected, dict(zip(tasks, map(int, quotas)))


def without_question_record(record):
    """Mirror PreSel/data_process/remove_instruction.py, including answer joining."""
    humans = [dict(x, value='<image>') for x in record['conversations']
              if x['from'] == 'human' and '<image>' in x['value']]
    if len(humans) != 1:
        raise ValueError('Expected one image-bearing turn for the official no-question transform')
    answers = [x['value'] for x in record['conversations'] if x['from'] == 'gpt']
    if not answers: raise ValueError('No assistant response')
    return {**record, 'conversations':humans+[{'from':'gpt', 'value':' '.join(answers)}]}


def importance_by_record(records, reference, scores):
    reference = set(reference)
    if len(scores) != len(reference) or {s['index'] for s in scores} != reference:
        raise ValueError('IRS must cover every reference record exactly once')
    ratios = defaultdict(list)
    for s in scores:
        numerator, denominator = float(s['loss_with_question']), float(s['loss_without_question'])
        if not math.isfinite(numerator) or not math.isfinite(denominator) or numerator < 0 or denominator <= 0:
            raise ValueError('Invalid IRS losses')
        ratios[records[s['index']]['task']].append(numerator/denominator)
    tasks = task_order(records)
    if set(ratios) != set(tasks): raise ValueError('Reference must represent every task')
    means = np.asarray([np.mean(ratios[t]) for t in tasks])
    temperature = 1/np.sqrt(len(tasks))
    logits = -means/temperature
    weights = np.exp(logits-logits.max()); weights /= weights.sum()
    return {t:{'mean_irs':float(m),'weight':float(w),'reference_records':len(ratios[t])}
            for t,m,w in zip(tasks,means,weights)}, float(temperature)


def task_quotas(importance, tasks, budget):
    weights = np.asarray([importance[t]['weight'] for t in tasks])
    quotas = (weights*budget).astype(int)
    quotas[int(np.argmax(weights))] += budget-int(quotas.sum())
    return dict(zip(tasks,map(int,quotas)))


def official_typical_selector(source_root=None):
    """Run the author's exact cluster quota and typicality functions.

    Load just the function definitions so unrelated optional upstream imports
    (ipdb, PCA, kmeans_pytorch) are not required. Their bodies stay unchanged.
    """
    if source_root is None:
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
        return typical_sample_selection
    import faiss
    source_root = Path(source_root)
    names = {'get_nn','get_mean_nn_similarity','calculate_typicality','typical_sample_selection'}
    nodes = []
    for path in (source_root/'data_process/clust_utils.py',source_root/'data_process/kmeans_clust.py'):
        tree = ast.parse(path.read_text())
        nodes.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    if {n.name for n in nodes} != names: raise ValueError('Missing author selection functions')
    module = ast.Module(body=nodes,type_ignores=[])
    scope = {'np':np,'faiss':faiss}
    exec(compile(ast.fix_missing_locations(module),'author_presel_functions','exec'),scope)
    return scope['typical_sample_selection']


def select_task(task, positions, features, reference, quota, source_root=None, seed=42):
    """Cluster the complete task's records, then exclude reference for NC selection."""
    positions = np.asarray(positions,dtype=np.int64)
    features = np.asarray(features,dtype=np.float32)
    if len(features) != len(positions) or not np.isfinite(features).all():
        raise ValueError('Feature matrix and record positions differ')
    norms = np.linalg.norm(features,axis=1,keepdims=True)
    if np.any(norms == 0): raise ValueError('Zero DINO features')
    features = features/norms
    clusters = max(1,int(len(positions)*0.01))
    # Same estimator/options as clust_utils.perform_clustering, except user seed.
    estimator = KMeans(n_clusters=clusters,random_state=seed)
    labels = estimator.fit_predict(features)
    selected = official_typical_selector(source_root)(positions,features,labels,quota,set(reference))
    if len(set(selected)) != len(selected) or set(selected)&set(reference):
        raise ValueError('Task selector emitted duplicates or reference records')
    return selected, labels, estimator.cluster_centers_


def combine_selection(records, reference, task_selections, seed=42):
    """Mirror data_selection.py: union record IDs + reference, random deficit fill."""
    initial = int(len(records)*0.05)
    if len(reference) != initial: raise ValueError('Reference must be the exact 5% budget')
    extra = [i for ids in task_selections for i in ids]
    if len(set(extra)) != len(extra) or set(extra)&set(reference):
        raise ValueError('Invalid selected record IDs')
    deficit = initial*2-len(extra)
    if deficit < 0: raise ValueError('Additional selection exceeds 10%')
    used = set(extra)|set(reference)
    if not used <= set(range(len(records))): raise ValueError('Record ID outside pool')
    fill = random.Random(seed).sample([i for i in range(len(records)) if i not in used],deficit)
    final = sorted([*reference,*extra,*fill])
    if len(final) != initial*3 or len(set(final)) != len(final) or not set(reference)<=set(final):
        raise ValueError('Final exact 15% / reference inclusion check failed')
    return final, fill


def source_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(root)/'data_process/kmeans_clust.py',Path(root)/'data_process/clust_utils.py',
                      Path(root)/'presel/llava_task_importance.py',Path(root)/'presel/data_selection.py',
                      Path(root)/'data_process/remove_instruction.py')}
