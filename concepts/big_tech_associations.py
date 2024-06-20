import numpy as np
from numpy.core.defchararray import join
from numpy.core.shape_base import _block_slicing
import pandas as pd
from os import path
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns

def process(top_100k_embeddings, result):
    embedding_df = pd.read_csv(top_100k_embeddings, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)

    # Get mean cosine similarities with Big Tech words

    big_tech_words = ['Google', 'Amazon', 'Facebook', 'Microsoft', 'Apple', 'Nvidia', 'Intel', 'IBM', 'Huawei',
                      'Samsung', 'Uber', 'Alibaba']

    big_tech_embs = embedding_df.loc[[word for word in big_tech_words if word in embedding_df.index]].to_numpy()
    big_tech_normed = big_tech_embs / np.linalg.norm(big_tech_embs, axis=-1, keepdims=True)

    all_embs = embedding_df.to_numpy()
    all_embs_normed = all_embs / np.linalg.norm(all_embs, axis=-1, keepdims=True)

    associations = all_embs_normed @ big_tech_normed.T
    means = np.mean(associations, axis=1)

    # Write dataframe to file

    big_tech_df = pd.DataFrame(means, index=embedding_df.index.tolist(), columns=['big_tech_es'])
    largest = big_tech_df.nlargest(10000, 'big_tech_es')

    largest.to_csv(result)

if __name__ == "__main__":
    # process("../raw/glove_100000_most_freq_skip.txt", "../results/six_methods/big_tech/big_tech_associations_glove.csv")
    # process("../raw/ft_100000_most_freq_skip.csv", "../results/fasttext/big_tech/big_tech_associations_ft.csv")
    # process("../openAI/openAI_100000_most_freq_skip.txt", "../results/openAI/big_tech/big_tech_associations_openAI.csv")
    # process("../cohere/cohere_100000_most_freq_skip.txt", "../results/cohere/big_tech/big_tech_associations_cohere.csv")
    process("../google/google_100000_most_freq_skip.txt", "../results/google/big_tech/big_tech_associations_google.csv")