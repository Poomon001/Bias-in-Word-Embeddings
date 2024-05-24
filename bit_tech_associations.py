import numpy as np
from numpy.core.defchararray import join
import pandas as pd
from os import path
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns

def process(embeddingInoutPath, resultOutputPath):
    embedding_df = pd.read_csv(embeddingInoutPath, sep=' ', header=None, index_col=0, na_values=None, skiprows=1, keep_default_na=False, quoting=csv.QUOTE_NONE)

    # Get mean cosine similarities with Big Tech words
    big_tech_words = ['Google', 'Amazon', 'Facebook', 'Microsoft', 'Apple', 'Nvidia', 'Intel', 'IBM', 'Huawei', 'Samsung', 'Uber', 'Alibaba']
    big_tech_words = big_tech_words + [name.upper() for name in big_tech_words]

    print(embedding_df.index)
    big_tech_embs = embedding_df.loc[[word for word in big_tech_words if word in embedding_df.index]].to_numpy()
    big_tech_normed = big_tech_embs / np.linalg.norm(big_tech_embs, axis=-1, keepdims=True)

    all_embs = embedding_df.to_numpy()
    all_embs_normed = all_embs / np.linalg.norm(all_embs,axis=-1,keepdims=True)

    print(big_tech_embs)
    print(big_tech_normed)

    # associations = all_embs_normed @ big_tech_normed.T
    # means = np.mean(associations, axis=1)
    #
    # # Write dataframe to file
    #
    # big_tech_df = pd.DataFrame(means, index=embedding_df.index.tolist(), columns=['big_tech_es'])
    # largest = big_tech_df.nlargest(10000, 'big_tech_es')
    #
    # largest.to_csv(resultOutputPath)

if __name__ == "__main__":
    process("results/sc-weat/openAI_words.csv", "results/big_tech/openAI_female_male_ratios.csv")