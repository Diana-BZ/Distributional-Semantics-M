#!/usr/bin/env python3

import sys
import nltk
from nltk.corpus import brown
import string

import gensim
from gensim.models.word2vec import Word2Vec
import scipy
from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import cosine
import numpy as np


"""Preprocessing Brown Corpus"""
PUNCTUATION = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-',
               '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^',
               '_', '`', '{', '|', '}', '~', '``', "''", '--']


def w2v_model(corpus, window):
    """Build Continuous Bag of Words (CBOW) model using Word2Vec

    Size, min_count, workers, and iterations are tuned
    """
    window_val = int(window)
    model = Word2Vec(sentences=corpus, window=window_val, size=35, min_count=1, workers=6, iter=100)
    return model


def sim_score(arr1, arr2):
    """Compute the cosine similarity
    """
    sim_score = cosine(arr1, arr2)
    return (1 - sim_score)


def spearman_correlation(my_sim_score, j_sim_score):
    spearman = spearmanr(my_sim_score, j_sim_score)
    return spearman


def preprocess(sentences):
    words = [[x.lower() for x in words if x not in PUNCTUATION] for words in sentences]
    return words


def main(window, judgment_filename, words):
    processed_words = preprocess(words)
    model = w2v_model(processed_words, window)
    with open(judgment_filename) as f:
        j_scores = []
        my_scores = []
        for line in f:
            line = line.strip()
            word1, word2, j_sim_score = line.split(',')
            wv1 = model.wv[word1]
            wv2 = model.wv[word2]
            j_scores += [float(j_sim_score)]
            my_sim = sim_score(wv1, wv2)
            my_scores += [my_sim]
            print(f"{word1},{word2}:{my_sim}")

        s_corr_coef = spearman_correlation(np.array(my_scores), np.array(j_scores))
        print(f"Correlation:{s_corr_coef[0]}")


if __name__ == '__main__':
    # Format: dist_sim.sh <window> <weighting> <judgment_filename>
    window = sys.argv[1]
    judgment_filename = sys.argv[2]
    corpus = brown.sents()
    main(window, judgment_filename, corpus)
