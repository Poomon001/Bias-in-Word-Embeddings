import numpy as np
import pandas as pd
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt

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

def process(top_100k_embeddings, top_50_university_embeddings, top_universities, output_weats, output_top_universities, pdf, group1_stimuli, group2_stimuli):
    print(top_100k_embeddings)

    # add top university names to the embedding_df for an analysis
    top_100k_embedding_df = pd.read_csv(top_100k_embeddings, sep=' ', header=None, index_col=0, na_values=None,
                                        keep_default_na=False, quoting=csv.QUOTE_NONE)
    top_50_embedding_df = pd.read_csv(top_50_university_embeddings, sep=',', header=0, index_col=0)
    top_100k_embedding_df.columns = top_50_embedding_df.columns

    # Ensure that the concatenation of top_50_embedding_df and top_100k_embedding_df maintains unique elements
    embedding_df = pd.concat([top_100k_embedding_df, top_50_embedding_df])

    group1_embeddings, group2_embeddings = embedding_df.loc[group1_stimuli].to_numpy(), embedding_df.loc[
        group2_stimuli].to_numpy()

    # Read in top university associations and take the words that are associated with top universities in both embeddings
    largest_models = []
    for top_university in top_universities:
        largest_model = pd.read_csv(top_university, index_col=0)
        largest_models.append(largest_model)

    # Initialize the set with indices from the first DataFrame
    common_indices = set(largest_models[0].index)

    # Iterate through the rest of the DataFrames and find the intersection
    for model in largest_models[1:]:
        common_indices.intersection_update(model.index)

    # Convert the set to a list if needed
    joint = list(common_indices)

    # Get group associations of top university words
    joint_vals = []

    for word in joint:
        joint_emb = embedding_df.loc[word].to_numpy()
        es, p = SC_WEAT(joint_emb, group1_embeddings, group2_embeddings, 1000)
        joint_vals.append([es, p])

    joint_arr = np.array(joint_vals)
    top_university = pd.DataFrame(joint_arr, index=joint, columns=[f'Effect_Size', 'P_Value'])
    top_university.to_csv(output_weats)

    print("Total Size:", len(top_university))

    # Write top university words to file
    words = top_university.index.tolist()
    with open(output_top_universities, 'w', encoding='utf-8') as writer:
        writer.write(', '.join(sorted(words, key=str.lower)))

    # Get percentage of top university words with minimum effect size
    es_list = [0, .2, .5, .8]

    pct_group1, pct_group2, actual_group1_counts, actual_group2_counts = [], [], [], []

    for es in es_list:
        group1_df = top_university[(top_university.Effect_Size >= es)]
        pct_group1.append(len(group1_df.index.tolist()) / len(top_university.index.tolist()))
        actual_group1_counts.append(len(group1_df.index.tolist()))

        group2_df = top_university[(top_university.Effect_Size <= -es)]
        pct_group2.append(len(group2_df.index.tolist()) / len(top_university.index.tolist()))
        actual_group2_counts.append(len(group2_df.index.tolist()))

    print(pct_group1)
    print(pct_group2)

    fig, ax = plt.subplots()

    # Plotting the results
    bar_width = 0.35
    index = range(len(es_list))

    bars1 = plt.bar(index, pct_group1, bar_width, label=f'{group1_stimuli[0].title()}')
    bars2 = plt.bar([i + bar_width for i in index], pct_group2, bar_width, label=f'{group2_stimuli[0].title()}')

    plt.xlabel('Effect Size')
    plt.ylabel('Percentage')
    plt.title(f'Percentage of {group2_stimuli[0].title()} and {group1_stimuli[0].title()} for Different Effect Sizes')
    plt.xticks([i + bar_width / 2 for i in index], es_list)
    plt.legend()

    # Adding numbers on top of the bars
    for i, bar in enumerate(bars1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), actual_group1_counts[i],
                ha='center', va='bottom', fontsize=8)

    for i, bar in enumerate(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), actual_group2_counts[i],
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save the plot as PDF
    plt.savefig(pdf)

    plt.show()

    print("\n === complete === \n")

if __name__ == "__main__":
    top_universities = []
    top_university1 = "../results/cohere/top_university/top_university_associations_cohere.csv"
    top_university2 = "../results/BGE/top_university/top_university_associations_BGE.csv"
    top_university3 = "../results/google/top_university/top_university_associations_google.csv"
    top_university4 = "../results/microsoft/top_university/top_university_associations_microsoft.csv"
    top_university5 = "../results/openAI/top_university/top_university_associations_openAI.csv"
    top_universities.append(top_university1)
    top_universities.append(top_university2)
    top_universities.append(top_university3)
    top_universities.append(top_university4)
    top_universities.append(top_university5)

    # Get gender stimuli
    female_stimuli = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
    male_stimuli = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']

    top_100k_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_50_university_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_top_50_universities.csv"
    output_weats = "../results/BGE/top_university/BGE_gender_top_university_weats.csv"
    output_top_universities = "../results/BGE/top_university/BGE_top_university_words.txt"
    pdf = "../plots/BGE/top_university/BGE_gender_top_university_ratio.pdf"
    process(top_100k_embeddings, top_50_university_embeddings, top_universities, output_weats, output_top_universities, pdf, female_stimuli, male_stimuli)

    top_100k_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_50_university_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_top_50_universities.csv"
    output_weats = "../results/cohere/top_university/cohere_gender_top_university_weats.csv"
    output_top_universities = "../results/cohere/top_university/cohere_top_university_words.txt"
    pdf = "../plots/cohere/top_university/cohere_gender_top_university_ratio.pdf"
    process(top_100k_embeddings, top_50_university_embeddings, top_universities, output_weats, output_top_universities, pdf, female_stimuli, male_stimuli)

    top_100k_embeddings = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_50_university_embeddings = "D:/Honour_Thesis_Data/google/google_top_50_universities.csv"
    output_weats = "../results/google/top_university/google_gender_top_university_weats.csv"
    output_top_universities = "../results/google/top_university/google_top_university_words.txt"
    pdf = "../plots/google/top_university/google_gender_top_university_ratio.pdf"
    process(top_100k_embeddings, top_50_university_embeddings, top_universities, output_weats, output_top_universities, pdf, female_stimuli, male_stimuli)

    top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_50_university_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_top_50_universities.csv"
    output_weats = "../results/microsoft/top_university/microsoft_gender_top_university_weats.csv"
    output_top_universities = "../results/microsoft/top_university/microsoft_top_university_words.txt"
    pdf = "../plots/microsoft/top_university/microsoft_gender_top_university_ratio.pdf"
    process(top_100k_embeddings, top_50_university_embeddings, top_universities, output_weats, output_top_universities, pdf, female_stimuli, male_stimuli)

    top_100k_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_50_university_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_top_50_universities.csv"
    output_weats = "../results/openAI/top_university/openAI_gender_top_university_weats.csv"
    output_top_universities = "../results/openAI/top_university/openAI_top_university_words.txt"
    pdf = "../plots/openAI/top_university/openAI_gender_top_university_ratio.pdf"
    process(top_100k_embeddings, top_50_university_embeddings, top_universities, output_weats, output_top_universities, pdf, female_stimuli, male_stimuli)