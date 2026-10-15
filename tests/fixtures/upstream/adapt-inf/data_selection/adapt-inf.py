import argparse
import json
import torch
import pickle
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
from collections import defaultdict, Counter
import os
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import sys
from sklearn.manifold import TSNE
import math


def semdedup(embeddings, cluster_labels, num_clusters, keep_samples=0.9, sort=False, cluster_centers=None, visualize=False):

    assert type(keep_samples) == int

    all_idxs = []
    all_Ms = []

    cluster_sizes = [np.where(cluster_labels == i)[0].shape[0] for i in range(num_clusters)]
    keep_budget_by_cluster = assign_budget_to_cluster(cluster_sizes, keep_samples, num_clusters)
    print("Budget for SemDeDup")
    for i, (old, new) in enumerate(zip(cluster_sizes, keep_budget_by_cluster)):
        print(f"cluster {i}: {old} --> {new}")

    for i in tqdm(range(num_clusters), desc="Running SemDeDup over clusters"):
    
        # print("Cluster: %s" % i)
        cluster_idxs = np.where(cluster_labels == i)[0]
        # print(cluster_idxs)
        cluster_i_embeddings = torch.tensor(embeddings[cluster_idxs]).type(torch.float32)
        cluster_i_embeddings = torch.nn.functional.normalize(cluster_i_embeddings, p=2, dim=-1)
        # dist_from_centroid = np.linalg.norm(embeddings[cluster_idxs] - cluster_centers[i], axis=-1)
        # sorted_idxs = np.argsort(dist_from_centroid)[::-1]
    
        # We use descending = True / False for keeping examples with low/ high similarity to cluster centroids . We ignore this step for keeping random examples from each group of similar examples . See Appendix D for more details about this step .
        # Compute the pairwise cosine similarity between embeddings
    
        pairwise_sim_matrix = cluster_i_embeddings @ cluster_i_embeddings.T
        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal = 1)
        M = torch.max(triu_sim_matrix, dim =0)[0]
        sorted_idxs = torch.argsort(M)
        points_to_keep = sorted_idxs[:keep_budget_by_cluster[i]].numpy()
        points_to_keep = cluster_idxs[points_to_keep]
        all_idxs.append(points_to_keep)
        all_Ms.append(M)
        # Check if the maximum similarity <= the threshold

    all_idxs = np.concatenate(all_idxs, axis=0)
    all_Ms = torch.cat(all_Ms, dim=0)

    print(f"Selected {points_to_keep.shape[0]} samples to retain after deduplication")
    return all_idxs, all_Ms.numpy()


def get_mwp_score(scores, selected_idxs):
    
    entropies = []
    score_keys = list(scores.keys())
    for k in score_keys:
        entropies.append(get_dist_entropy(scores[k][selected_idxs]))
    for k, e in zip(score_keys, entropies):
        print(f"{k}: {e}")
    print("Selecting ", score_keys[np.argmax(entropies)])
    return scores[score_keys[np.argmax(entropies)]][selected_idxs]


def get_dist_entropy(x, nbins=50, verbose=False):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """

    # normalize with mean and std dev
    nan_count = np.count_nonzero(np.isnan(x))
    if verbose:
        x = x.astype(np.float32)
        print("Replacing %s nan values in input vector" % nan_count)
        print(x, np.mean(x), np.std(x))
    x = np.nan_to_num(x)
    
    x_std = np.std(x)
    x = (x - np.mean(x))/x_std

    counts = np.zeros(nbins)
    bins = np.linspace(-4*x_std, 4*x_std, nbins)
    if verbose:
        print(bins)
        print(x)
    for i in range(1, nbins):
        counts[i] = np.sum((x > bins[i-1]) & (x <= bins[i]))

    epsilon = np.finfo(float).eps
    # divide by number of columns to get the probability of each unique value
    p = counts / np.sum(counts)
    # Replace zeros with epsilon
    if verbose:
        print(p)
    p = np.where(p == 0, epsilon, p)
    if verbose:
        print(p)
    nan_count = np.count_nonzero(np.isnan(p))
    if verbose:
        print("Replacing %s nan values in prob. vector" % nan_count)
    p = np.nan_to_num(p)
    if verbose:
        print(p)
        print(np.log2(p))

    # compute Shannon entropy in bits
    return -np.sum(p * np.log2(p))


def get_model_suffix(model_name_param):
    return model_name_param.split('/')[-1].replace('.pkl', '')


def calc_prototypicality_scores(X, labels, cluster_centers):
    return np.linalg.norm(X - cluster_centers[labels], axis=-1)

def calc_norm_scores(X):
    return np.linalg.norm(X, axis=-1)

def save_scores(file_param, data_label_dict, scores, suffix='proto'):

    files = file_param.split(',')
    assert len(files) == len(data_label_dict.keys())

    for f, k in zip(files, list(data_label_dict.keys())):
        start_idx, stop_idx = data_label_dict[k]
        try:
            data = json.load(open(f.replace('.pt', '.json')))
        except:
            data = json.load(open(f.replace('_dim=0.pt', '.json')))
        assert len(data) == stop_idx - start_idx
        for i in range(len(data)):
            data[i][suffix] = scores[start_idx:stop_idx][i]
        with open(f.replace('.pt', '.json'), 'w') as fout:
            json.dump(data, fout, indent=2)
        # np.save(f.replace('.pt', '_%s.npy' % suffix), scores[start_idx:stop_idx])


def get_importance_scores(file_param):
    files = file_param.split(',')
    scores = []
    ppls = []
    ids = []
    for f in files:
        print("Reading from json of %s" % f)
        data = json.load(open(f.replace('.pt', '.json')))
        scores.extend([-d['ppl'][1] if int(d['ppl'][0]) == -1 else d['ppl'][0]/d['ppl'][1] for d in data])
        ppls.extend([d['ppl'][0] for d in data])
        ids.extend([d['id'] for d in data])
    return np.array(scores), np.array(ppls), ids


def read_data(file_param, return_dict=True):
    files = file_param.split(',')

    if return_dict:
        id2data = {}
        for f in files:
            print("Reading from json of %s" % f)
            data = json.load(open(f))
            for d in data:
                d["dataset"] ="m3it" if "m3it" in f else "llava"
                id2data[d["id"]] = d
        return id2data
    else:
        data = []
        for f in files:
            dataset = os.path.split(f)[-1]
            subset = json.load(open(f))
            for i, d in enumerate(subset):
                subset[i]['dataset'] = dataset
            data.extend(subset.copy())
            print(f"Read {len(subset)} samples from {f}")
        return data


def get_label(string):

    if string.startswith('llava'):
        return 'llava'
    elif string.startswith('visionflan'):
        return 'visionflan'
    elif string.startswith('lamm'):
        return 'lamm'
    elif string.startswith('m3it'):
        return 'm3it'
    elif 'minigpt4' in string or string.startswith('cc_sbu_align'):
        return 'minigpt4'
    else:
        return 'm3it'
        # raise ValueError
    

def load(args):
    # load
    with open(args.model_file, 'rb') as f:
        clf = pickle.load(f)
    return clf


def save(clf, args):
    # save
    with open(args.model_file,'wb') as f:
        pickle.dump(clf, f)


def cluster(X, args):

    start_time = time.time()

    mbkm = MiniBatchKMeans(n_clusters=args.n_clusters, batch_size=10240)  # Take a good look at the docstring and set options here
    mbkm.fit(X)
    print(f'Finished KMeans on {X.shape[0]} samples in Time (seconds): {time.time() - start_time}')
    return mbkm


def load_embeddings(param_string):

    if ',' not in param_string:
        X = torch.nn.functional.normalize(torch.load(param_string).detach(), dim=-1).numpy()
        label_dict = {get_label(os.path.split(param_string)[-1]): [0, X.shape[0]]}
    else:
        filenames = param_string.split(',')
        embeddings = []
        label_dict = {}
        offset = 0
        for f in filenames:
            embeddings.append(torch.load(f).detach().numpy())
            label_dict[get_label(os.path.split(f)[-1])] = [offset, offset + embeddings[-1].shape[0]]
            offset += embeddings[-1].shape[0]
        X = np.concatenate(embeddings, axis=0)

    return X, label_dict


def load_scores(param_string):

    if ',' not in param_string:
        all_scores = torch.load(param_string)
        for k in all_scores.keys():
            all_scores[k] = all_scores[k].squeeze().cpu().numpy()
        
    else:
        filenames = param_string.split(',')
        all_scores = []
        for f in filenames:
            all_scores.append(torch.load(f))
            print(f"Read {all_scores[-1]['ppl'].shape[0]} sscores from {f}")
        all_scores = {k: torch.cat([scores[k].squeeze().detach().cpu() for scores in all_scores]).numpy() for k in all_scores[0].keys()}

    if 'ppl' in all_scores:
        all_scores['ppl'] = np.log(all_scores['ppl'])
    if 'grand' in all_scores:
        all_scores['grand'] = all_scores['grand'].astype(np.float32) # to prevent overflow and nan errors

    return all_scores


def eval(clf, X):
    return clf.predict(X)


def set_outlier_threshold(scores):

    scores = np.sort(scores)
    low_thresh = scores[int(len(scores)*0.05)]
    high_thresh = scores[int(len(scores)*0.95)]
    print("Setting outlier threshold to %s (low) and %s (high)" % (low_thresh, high_thresh))
    return low_thresh, high_thresh
    

def get_ccs_samples(scores, bins, budget):

    # low, high = set_outlier_threshold(list(scores.values()))
    low, high = set_outlier_threshold(scores)
    right = high

    remaining = budget
    budget = math.ceil(budget/bins)

    selected_idxs = []
    for i in tqdm(range(bins), desc="Getting CCS samples"):
        left = right - ((high-low)/bins)
        # candidates = [d for d in data.keys() if d in scores and scores[d] <= right and scores[d] >= left]
        candidates = [j for j, score in enumerate(scores) if score > left and score <= right]
        if len(candidates) < budget:
            selected_idxs.extend(candidates)
            if (i+1) < bins:
                budget = math.ceil((remaining - len(candidates))/(bins-(i+1)))
                remaining -= len(candidates)
        else:
            selected_idxs.extend(random.sample(candidates, k=budget))
            remaining -= budget
        print("Selected %s samples from between %s and %s values" % (len(selected_idxs), left, right))
        right = left
    
    return np.array(selected_idxs)


def calc_wss_sil(embeddings, labels, centroids):

    n_clusters = np.max(labels) + 1
    wsss = []
    cluster_sizes = []
    for i in range(0, n_clusters):
        X = embeddings[labels == i] - centroids[i]
        cluster_sizes.append(X.shape[0])
        wsss.append(np.linalg.norm(X, ord=2, axis=-1).sum())
    # sil = silhouette_score(embeddings, labels, metric = 'euclidean')
    print("Cluster sizes: ", cluster_sizes)
    print("WSSS per cluster: ", wsss)
    return np.sum(wsss), 0


def get_dbp_probs(centroids, embeddings, labels):

    nan_count = np.count_nonzero(np.isnan(centroids))
    print("Replacing %s nan values in centroids" % nan_count)
    centroids = np.nan_to_num(centroids)

    nan_count = np.count_nonzero(np.isnan(embeddings))
    print("Replacing %s nan values in embeddings" % nan_count)
    embeddings = np.nan_to_num(embeddings)

    probs = []
    for i in range(args.n_clusters):

        cluster_idxs = list(np.where(labels == i))[0]
        centroid = centroids[i]
        # cluster_scores = scores[cluster_idxs]

        mean_dist_from_centroid = np.mean(np.linalg.norm(embeddings[cluster_idxs] - centroid, axis=-1))
        mean_dist_from_clusters = np.mean(np.linalg.norm(centroids-centroid))
        probs.append(mean_dist_from_centroid*mean_dist_from_clusters)

        # probs.append(np.mean(cluster_scores))
    
    print("Un-normalized complexity scores: ", probs)
    probs = [n/sum(probs) for n in probs]
    return probs


def assign_budget_to_cluster(cluster_sizes, total_budget, n_clusters, score_weighted=False, el2n_scores=None, labels=None):

    if score_weighted:
        el2n_scores = np.nan_to_num(el2n_scores)
        cluster_scores = []
        for i in range(n_clusters):
            cluster_idxs = np.where(labels==i)[0]
            cluster_scores.append(el2n_scores[cluster_idxs])
        cluster_probs = [np.std(scores) for scores in cluster_scores]
        cluster_probs = [std/sum(cluster_probs) for std in cluster_probs]
        print('Initial prob. assignment', cluster_probs)
        cluster_budgets = [int(p * total_budget) for p in cluster_probs]
        print('Initial budget assignment', cluster_budgets)
        sorted_cluster_idxs = np.argsort(cluster_sizes)
        for i, idx in enumerate(sorted_cluster_idxs):
            print(f"Processing for cluster {idx}")
            curr_cluster_size = cluster_sizes[idx]
            p = cluster_probs[idx]
            if curr_cluster_size < cluster_budgets[idx]:
                adjusted_p = curr_cluster_size/total_budget
                remaining_p = (p - adjusted_p)/(n_clusters-i-1)
                for next_idx in sorted_cluster_idxs[i+1:]:
                    cluster_probs[next_idx] += remaining_p
                    cluster_budgets[next_idx] = int(cluster_probs[next_idx] * total_budget)
                cluster_budgets[idx] = curr_cluster_size
                print("Updated prob. assignment: ", cluster_probs)
            else:
                continue
        # cluster_budgets = [cluster_budgets[idx] for idx in np.argsort(sorted_cluster_idxs)]
        return cluster_budgets            
                    
    else:

        smallest_cluster_size = np.min(cluster_sizes)
        if smallest_cluster_size*n_clusters >= total_budget:
            print("Reducing budget from smallest cluster size %s to fit %s" % (smallest_cluster_size, total_budget))
            smallest_cluster_size = math.ceil(total_budget/args.n_clusters)
            return [smallest_cluster_size]*n_clusters
        else:
            budgets_per_bin = []
            sorted_cluster_idxs = np.argsort(cluster_sizes)
            budget_per_bin = math.ceil(total_budget/n_clusters)
            for i in range(n_clusters):
                curr_cluster_size = cluster_sizes[sorted_cluster_idxs[i]] 
                if curr_cluster_size < budget_per_bin:
                    budgets_per_bin.append(curr_cluster_size)
                    budget_per_bin = budget_per_bin + math.ceil((budget_per_bin-curr_cluster_size)/(n_clusters-i-1))
                else:
                    budgets_per_bin.append(budget_per_bin)
            budgets_per_bin = [budgets_per_bin[idx] for idx in np.argsort(sorted_cluster_idxs)]
            return budgets_per_bin


def main(args):


    if args.kmeans:

        assert (args.train_emb_file and args.model_file) or (args.model_file and args.eval_emb_file)

        if args.train_emb_file:
            X_train, X_train_label_dict = load_embeddings(args.train_emb_file)
        else:
            X_train = None
        
        if args.overwrite or (not os.path.exists(args.model_file) and X_train is not None):
            print("Training on %s train samples" % X_train.shape[0])
            nan_count = np.count_nonzero(np.isnan(X_train))
            print("Replacing %s nan values" % nan_count)
            X_train = np.nan_to_num(X_train)
            clf = cluster(X_train, args)
            save(clf, args)
        else:
            clf = load(args)
        
        X_train_labels = clf.labels_
        # print(X_train_labels[:10])
        wss, sil = calc_wss_sil(X_train, X_train_labels, clf.cluster_centers_)
        print("Num. clusters: %s, WSS: %.2f, Silhouette score: %.2f" % (args.n_clusters, wss, sil))

    selected_samples = []
    
    if args.semdedup:
        all_scores = load_scores(args.score_file)
        if args.semdedup_emb_file:
            semdedup_embeddings, _ = load_embeddings(args.semdedup_emb_file)
            selected_idxs, pairwise_sims = semdedup(semdedup_embeddings, X_train_labels, args.n_clusters, args.data_budget)
        else:
            selected_idxs, pairwise_sims = semdedup(X_train, X_train_labels, args.n_clusters, args.data_budget)

        data = read_data(args.data_file, return_dict=False)

        for k in all_scores.keys():
            all_scores[k] = all_scores[k][selected_idxs]

        X_train_labels = X_train_labels[selected_idxs]
        cluster_sizes = []
        for i in range(0, args.n_clusters):
            cluster_sizes.append(np.sum(X_train_labels==i))
        print("Cluster sizes after DeDup: ", cluster_sizes)
        data = [data[idx] for idx in selected_idxs]


    if args.select:
        selected_samples = []
        if not args.visualize and not args.semdedup:
            data = read_data(args.data_file, return_dict=False)
            if args.score_type or args.mwp:
                all_scores = load_scores(args.score_file)
                for k in all_scores.keys():
                    all_scores[k] = all_scores[k].squeeze()

        assert args.score_type or args.mwp or args.random

        if args.kmeans:

            assert X_train.shape[0] == len(data)
            assert all_scores['ppl'].shape[0] == len(data)

            for i in range(args.n_clusters):
                cluster_idxs = np.where(X_train_labels==i)[0].tolist()
                print(f" *****************Cluster {i} samples ******************")
                for idx in random.sample(cluster_idxs, k=10):
                    print(data[idx])

            if args.dbp:
                cluster_probs =  get_dbp_probs(clf.cluster_centers_, X_train, X_train_labels)
                cluster_budgets = [int(p*args.sample_budget) for p in cluster_probs]
            elif args.score_weighted:
                cluster_budgets = assign_budget_to_cluster([np.where(X_train_labels==i)[0].shape[0] for i in range(args.n_clusters)], 
                                                        args.sample_budget, args.n_clusters, score_weighted=True, el2n_scores=all_scores['el2n'], labels=X_train_labels)
            else:
                cluster_budgets = assign_budget_to_cluster([np.where(X_train_labels==i)[0].shape[0] for i in range(args.n_clusters)], 
                                                            args.sample_budget, args.n_clusters)
            
            print("Cluster budgets: ", cluster_budgets, "; Sum total =", sum(cluster_budgets))
            for i in range(args.n_clusters):
                print("Cluster ", i)
                cluster_idxs = np.where(X_train_labels==i)[0]
                if args.mwp:
                    cluster_scores = get_mwp_score(all_scores, cluster_idxs)
                elif args.score_type:
                    cluster_scores = all_scores[args.score_type][cluster_idxs]
                else:
                    pass

                if args.ccs:
                    selected_idxs = cluster_idxs[get_ccs_samples(cluster_scores, 50, cluster_budgets[i])]
                elif args.mid:
                    start = abs(int(cluster_scores.shape[0]//2) - int(cluster_budgets[i]//2))
                    end = -start
                    selected_idxs = cluster_idxs[np.argsort(cluster_scores)[start:end]]
                elif args.random:
                    # selected_idxs = np.random.choice(scores.shape[0], args.sample_budget)
                    selected_idxs = np.random.choice(cluster_idxs, cluster_budgets[i])
                else:
                    selected_idxs = cluster_idxs[np.argsort(cluster_scores)[:cluster_budgets[i]]] #min
                selected_samples.extend([data[idx] for idx in selected_idxs])

        else:

            if args.score_type:
                scores = all_scores[args.score_type]
            else:
                pass

            if args.ccs:
                selected_idxs = get_ccs_samples(scores, 50, args.sample_budget)
            elif args.mid:
                start = abs(int(scores.shape[0]//2) - int(args.sample_budget//2))
                end = -start
                selected_idxs = np.argsort(scores)[start:end]
            elif args.random:
                # selected_idxs = np.random.choice(scores.shape[0], args.sample_budget)
                selected_idxs = random.sample(list(range(len(data))), args.sample_budget)
            else:
                selected_idxs = np.argsort(scores)[:args.sample_budget] #min
            selected_samples = [data[idx] for idx in selected_idxs]


    def _process_img_path(img_path, dataset, pre_mantis=False):
        if pre_mantis:

            if dataset == 'm3it':
                new_img_path = os.path.join('m3it', img_path) if not img_path.startswith('m3it') else img_path
            elif dataset == 'cc_sbu_align':
                new_img_path = os.path.join('cc_sbu_align', img_path) if not img_path.startswith('cc_sbu_align') else img_path
            else:
                new_img_path = img_path

        else:
            mantis_prefix = '/nas-hdd/tarbucket/adyasha/'
            if dataset == 'visionflan':
                new_img_path = os.path.join('datasets', 'llava', 'visionflan', 'images_191task_1k', img_path)
            elif dataset == 'm3it':
                new_img_path = os.path.join('m3it', img_path) if not img_path.startswith('m3it') else img_path
                new_img_path = os.path.join('datasets', 'llava', new_img_path)
            elif dataset == 'cc_sbu_align':
                new_img_path = os.path.join('cc_sbu_align', img_path) if not img_path.startswith('cc_sbu_align') else img_path
                new_img_path = os.path.join('datasets', 'llava', new_img_path)
            elif dataset == 'mantis':
                new_img_path = img_path[len(mantis_prefix):] if img_path.startswith(mantis_prefix) else img_path
            elif dataset == 'lamm':
                new_img_path = os.path.join('datasets', 'llava', img_path)
            elif dataset == 'llava':
                new_img_path = os.path.join('datasets', 'llava', img_path)
            else:
                new_img_path = img_path

        return new_img_path

    
    if len(selected_samples) > 0:
        new_samples = []
        for i in tqdm(range(len(selected_samples))):
            new_samples.append(selected_samples[i])

        if args.score_type:
            args.out_file = args.out_file.replace('.json', f"-{args.score_type}.json")
        elif args.mwp:
            args.out_file = args.out_file.replace('.json', "-mwp.json")
        elif args.random:
            pass
        else:
            raise ValueError

        with open(args.out_file, 'w') as f:
            json.dump(new_samples, f, indent=2)
        print(f"Saved {len(new_samples)} samples to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-emb-file", type=str, default="")
    parser.add_argument("--semdedup-emb-file", type=str, default="")
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--model-file", type=str, default="")
    parser.add_argument("--score-file", type=str, default="")
    parser.add_argument("--eval-emb-file", type=str, default="")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--data-budget", type=int, default=1.0)
    parser.add_argument("--sample-budget", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out-file", type=str, default="")
    parser.add_argument("--score-type", type=str, default="")
    parser.add_argument("--mwp", action="store_true")
    parser.add_argument("--dbp", action="store_true")
    parser.add_argument("--score-weighted", action="store_true")
    parser.add_argument("--variance", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--semdedup", action="store_true")
    parser.add_argument("--ccs", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--kmeans", action="store_true")
    parser.add_argument("--mid", action="store_true")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    main(args)