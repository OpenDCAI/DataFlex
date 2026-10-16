# Vendored unmodified function bodies from the upstream adapt-inf source.
# MIT License
#
# Copyright (c) 2024 Adyasha Maharana
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

import math
import random

import numpy as np
import torch
from tqdm import tqdm

def obtain_entropy_score(outputs, eval_batch_size=None):
    # Entropy of the probability vector

    probs = torch.nn.Softmax(dim=-1)(outputs.logits)
    entropy = -1 * probs * torch.log(probs + 1e-10)
    # print(probs.shape, entropy.shape)
    entropy = torch.mean(torch.sum(entropy, dim=-1), dim=-1).detach().cpu().squeeze()
    # print('entropy', entropy.shape, entropy)
    return entropy

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


def obtain_ig_score(model, batch, ppl_with_image, eval_batch_size): #TODO: Fix

    if eval_batch_size == 1:
        batch['images'] = torch.zeros_like(batch['images'])
        # print(batch.keys())
        with torch.inference_mode():
            loss_no_image = model(**batch).loss
        ppl_wo_image = torch.exp(loss_no_image)
        ig_score = ppl_wo_image/ppl_with_image
        # print('ig_score', ig_score.shape, ig_score)
        return ig_score
    else:
        raise NotImplementedError


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
