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

def SC_WEAT(w, A, B, permutations):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T
    joint_associations = np.concatenate((A_associations,B_associations),axis=-1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations,ddof=1)

    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:,:midpoint],axis=1) - np.mean(sample_distribution[:,midpoint:],axis=1)
    p_value = 1 - norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1))

    return effect_size, p_value

def process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf):
    print(top_100k_embeddings)
    embedding_df = pd.read_csv(top_100k_embeddings, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)

    female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()

    # Read in big tech associations and take the words that are associated with big tech in both embeddings
    largest_models = []
    for bigtech in largestBigTechs:
        largest_model = pd.read_csv(bigtech, index_col=0)
        largest_models.append(largest_model)

    # Initialize the set with indices from the first DataFrame
    common_indices = set(largest_models[0].index)

    # Iterate through the rest of the DataFrames and find the intersection
    for model in largest_models[1:]:
        common_indices.intersection_update(model.index)

    # Convert the set to a list if needed
    joint = list(common_indices)

    # Get gender associations of big tech words
    joint_vals = []

    for word in joint:
        joint_emb = embedding_df.loc[word].to_numpy()
        es, p = SC_WEAT(joint_emb, female_embeddings, male_embeddings, 1000)
        joint_vals.append([es, p])

    joint_arr = np.array(joint_vals)
    big_tech = pd.DataFrame(joint_arr, index=joint, columns=['Effect_Size', 'P_Value'])
    big_tech.to_csv(output_weats)

    # Write big tech words to file
    words = big_tech.index.tolist()
    with open(output_bigtechs, 'w', encoding='utf-8') as writer:
        writer.write(', '.join(sorted(words, key=str.lower)))

    # Get percentage of big tech words with minimum gender effect size

    es_list = [0, .2, .5, .8]

    pct_female, pct_male = [], []

    for es in es_list:
        print(es)
        female_df = big_tech[(big_tech.Effect_Size >= es)]
        pct_female.append(len(female_df.index.tolist()) / len(big_tech.index.tolist()))

        male_df = big_tech[(big_tech.Effect_Size <= -es)]
        pct_male.append(len(male_df.index.tolist()) / len(big_tech.index.tolist()))

    print(pct_female)
    print(pct_male)

    fig, ax = plt.subplots()

    # Plotting the results
    bar_width = 0.35
    index = range(len(es_list))

    bars1 = plt.bar(index, pct_female, bar_width, label='Female')
    bars2 = plt.bar([i + bar_width for i in index], pct_male, bar_width, label='Male')

    plt.xlabel('Effect Size')
    plt.ylabel('Percentage')
    plt.title('Percentage of Male and Female for Different Effect Sizes')
    plt.xticks([i + bar_width / 2 for i in index], es_list)
    plt.legend()

    plt.tight_layout()

    # Save the plot as PDF
    plt.savefig(pdf)

    plt.show()

    print("\n === complete === \n")


if __name__ == "__main__":
    largestBigTechs = []
    bigtech1 = "../results/six_methods/big_tech/big_tech_associations_glove.csv"
    bigtech2 = "../results/fasttext/big_tech/big_tech_associations_ft.csv"
    bigtech3 = "../results/openAI/big_tech/big_tech_associations_openAI.csv"
    bigtech4 = "../results/cohere/big_tech/big_tech_associations_cohere.csv"
    bigtech5 = "../results/google/big_tech/big_tech_associations_google.csv"
    bigtech6 = "../results/microsoft/big_tech/big_tech_associations_microsoft.csv"
    bigtech7 = "../results/microsoft_norm/big_tech/big_tech_associations_microsoft_norm.csv"
    bigtech8 = "../results/BGE/big_tech/big_tech_associations_BGE.csv"
    largestBigTechs.append(bigtech1)
    largestBigTechs.append(bigtech2)
    largestBigTechs.append(bigtech3)
    largestBigTechs.append(bigtech4)
    largestBigTechs.append(bigtech5)
    largestBigTechs.append(bigtech6)
    largestBigTechs.append(bigtech7)
    largestBigTechs.append(bigtech8)

    top_100k_embeddings = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    output_weats = "../results/six_methods/big_tech/glove_big_tech_weats.csv"
    output_bigtechs = "../results/six_methods/big_tech/glove_big_tech_words.txt"
    pdf = "../plots/six_methods/bigtech/glove_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    output_weats = "../results/fasttext/big_tech/ft_big_tech_weats.csv"
    output_bigtechs = "../results/fasttext/big_tech/ft_big_tech_words.txt"
    pdf = "../plots/fasttext/bigtech/ft_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    output_weats = "../results/openAI/big_tech/openai_big_tech_weats.csv"
    output_bigtechs = "../results/openAI/big_tech/openai_big_tech_words.txt"
    pdf = "../plots/openAI/bigtech/openai_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    output_weats = "../results/cohere/big_tech/cohere_big_tech_weats.csv"
    output_bigtechs = "../results/cohere/big_tech/cohere_big_tech_words.txt"
    pdf = "../plots/cohere/bigtech/cohere_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    output_weats = "../results/google/big_tech/google_big_tech_weats.csv"
    output_bigtechs = "../results/google/big_tech/google_big_tech_words.txt"
    pdf = "../plots/google/bigtech/google_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    output_weats = "../results/microsoft/big_tech/microsoft_big_tech_weats.csv"
    output_bigtechs = "../results/microsoft/big_tech/microsoft_big_tech_words.txt"
    pdf = "../plots/microsoft/bigtech/microsoft_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft_norm/microsoft_norm_100000_most_freq_skip.txt"
    output_weats = "../results/microsoft_norm/big_tech/microsoft_norm_big_tech_weats.csv"
    output_bigtechs = "../results/microsoft_norm/big_tech/microsoft_norm_big_tech_words.txt"
    pdf = "../plots/microsoft_norm/bigtech/microsoft_norm_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)

    top_100k_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    output_weats = "../results/BGE/big_tech/BGE_big_tech_weats.csv"
    output_bigtechs = "../results/BGE/big_tech/BGE_big_tech_words.txt"
    pdf = "../plots/BGE/bigtech/BGE_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf)
