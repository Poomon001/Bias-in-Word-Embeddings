import numpy as np
from numpy.core.defchararray import join
from numpy.core.shape_base import _block_slicing
import pandas as pd
from os import path
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns

#Get gender stimuli
female_stimuli = ['female','woman','girl','sister','she','her','hers','daughter']
male_stimuli = ['male','man','boy','brother','he','him','his','son']

def process(top_100k_file, output):
    embedding_df = pd.read_csv(top_100k_file, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)

    female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()

    # Read in big tech associations and take the words that are associated with big tech in both embeddings
    # largest_ft = pd.read_csv(path.join(EMB_DIR, f'big_tech_associations_ft.csv'), index_col=0)
    largest_glove = pd.read_csv("results/six_methods/big_tech/big_tech_associations_glove.csv", index_col=0)
    joint = [i for i in largest_ft.index.tolist() if i in largest_glove.index]


if __name__ == "__main__":

    process("../raw/glove_100000_most_freq_skip.txt", "../results/six_methods/big_tech/big_tech_associations_ft.csv")