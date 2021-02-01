#!/usr/bin/env python3

import sys
import nltk
from nltk.corpus import brown
import string
import numpy as np
import scipy
from scipy.spatial.distance import cosine
from scipy.stats.mstats import spearmanr
from collections import defaultdict
from scipy.sparse import csr_matrix
from math import log2

word2idx = {}

"""Preprocessing Brown Corpus"""
PUNCTUATION = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-',
               '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^',
               '_', '`', '{', '|', '}', '~', '``', "''", '--']

def create_vector(words, window):
    """Vector Representations. Returns a sparse row matrix.
    """
    global word2idx
    window = int(window)
    word2idx = {word: idx for idx, word in enumerate(sorted(set(words)))}
    context_count = defaultdict(int)

    for i in range(len(words)):
        upper_bound = min(len(words), i + window)
        lower_bound = max(0, i-window)
        neighbors = words[lower_bound:i]
        if i < len(words) - 1:
            neighbors += words[i+1:upper_bound]
        word_idx = word2idx[words[i]]
        for n in neighbors:
            n_idx = word2idx[n]
            context_count[(word_idx, n_idx)] += 1
    left_indices = []
    right_indices = []
    values = []
    for key, value in context_count.items():
        l, r = key
        left_indices.append(l)
        right_indices.append(r)
        values.append(value)

    matrix = csr_matrix((values, (left_indices, right_indices)), shape=(len(word2idx), len(word2idx)))

    return matrix


def calculate_ppmi_all(matrix):
    """Calculate PPMI for matrix.
    """
    total_count = matrix.sum()
    w_total = np.array(matrix.sum(axis=1))
    c_total = np.array(matrix.sum(axis=0))

    prob_w = w_total/total_count
    prob_c = c_total/total_count
    prob_wc = matrix/total_count

    pmi = prob_wc / (prob_w @ prob_c)
    pmi =  np.log2(pmi)
    pmi = np.where(pmi < 0, np.zeros(pmi.shape), pmi)

    return pmi


def calculate_ppmi(matrix, widx1):
    total_count = matrix.sum()
    arr = np.array(matrix[widx1])
    prob_wc = arr / total_count
    p_w = arr.sum() / total_count
    p_c = matrix.sum(axis=0) / total_count

    pmi = np.log2(prob_wc) - np.log2(p_w * p_c)
    pmi = np.where(pmi < 0, np.zeros(pmi.shape), pmi)

    return pmi


def sim_score(arr1, arr2):
    """Compute the cosine similarity.
    Returns spearman correlation.
    """
    sim_score = cosine(arr1, arr2)
    return (1 - sim_score)


def spearman_correlation(my_sim_score, j_sim_score):
    spearman = spearmanr(my_sim_score, j_sim_score)
    return spearman


def get_top_feats(arr, top=10):
    """Get the top ten feats
    """
    global word2idx
    top_idx = np.argsort(arr)[-top:]
    idx2word = list(word2idx.keys())
    weights = arr[top_idx]
    words = [idx2word[i] for i in top_idx]
    # Search for 10 highest weights in array
    # Return features, weights
    return words, weights


def preprocess(words):
    words = [x.lower() for x in words if x not in PUNCTUATION]
    return words


def main(window, weighting, judgment_filename, words):
    processed_words = preprocess(words)
    matrix = create_vector(processed_words, window)
    if weighting == 'PMI':
        matrix = calculate_ppmi_all(matrix)
    else:
        matrix = matrix.toarray()
    global word2idx

    with open(judgment_filename) as f:
        j_scores = []
        my_scores = []
        for line in f:
            line = line.strip()
            word1, word2, j_sim_score = line.split(',')
            widx1 = word2idx[word1]
            widx2 = word2idx[word2]
            w_vec_1 = matrix[widx1]
            w_vec_2 = matrix[widx2]

            top_10_words, top_10_weights = get_top_feats(w_vec_1)
            out = [word1 + ':'] + [":".join(wrd_weight) for wrd_weight in zip(top_10_words, top_10_weights.astype(str))]
            print(*out)

            top_10_words, top_10_weights = get_top_feats(w_vec_2)
            out = [word2 + ':'] + [":".join(wrd_weight) for wrd_weight in zip(top_10_words, top_10_weights.astype(str))]
            print(*out)  # print each element of out with whitespace in-between

            j_scores += [float(j_sim_score)]
            my_score = sim_score(w_vec_1, w_vec_2)
            my_scores += [my_score]

            print(f"{word1},{word2}:{my_score}")

        s_corr_coef = spearman_correlation(np.array(my_scores), np.array(j_scores))
        print(f"Correlation:{s_corr_coef[0]}")


if __name__ == '__main__':
    # Format: dist_sim.py <window> <weighting> <judgment_filename>
    window = sys.argv[1]
    weighting = sys.argv[2]
    judgment_filename = sys.argv[3]
    main(window, weighting, judgment_filename, brown.words())
