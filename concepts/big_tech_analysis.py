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

def process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, group1_stimuli, group2_stimuli):
    print(top_100k_embeddings)
    embedding_df = pd.read_csv(top_100k_embeddings, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)

    group1_embeddings, group2_embeddings = embedding_df.loc[group1_stimuli].to_numpy(), embedding_df.loc[group2_stimuli].to_numpy()

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
        es, p = SC_WEAT(joint_emb, group1_embeddings, group2_embeddings, 1000)
        joint_vals.append([es, p])

    joint_arr = np.array(joint_vals)
    big_tech = pd.DataFrame(joint_arr, index=joint, columns=[f'Effect_Size', 'P_Value'])
    big_tech.to_csv(output_weats)

    print("Total Size:", len(big_tech))

    # Write big tech words to file
    words = big_tech.index.tolist()
    with open(output_bigtechs, 'w', encoding='utf-8') as writer:
        writer.write(', '.join(sorted(words, key=str.lower)))

    # Get percentage of big tech words with minimum gender effect size

    es_list = [0, .2, .5, .8]

    pct_group1, pct_group2, actual_group1_counts, actual_group2_counts = [], [], [], []

    for es in es_list:
        group1_df = big_tech[(big_tech.Effect_Size >= es)]
        pct_group1.append(len(group1_df.index.tolist()) / len(big_tech.index.tolist()))
        actual_group1_counts.append(len(group1_df.index.tolist()))

        group2_df = big_tech[(big_tech.Effect_Size <= -es)]
        pct_group2.append(len(group2_df.index.tolist()) / len(big_tech.index.tolist()))
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
    largestBigTechs = []
    bigtech1 = "../results/glove/big_tech/big_tech_associations_glove.csv"
    bigtech2 = "../results/fasttext/big_tech/big_tech_associations_ft.csv"
    bigtech3 = "../results/openAI/big_tech/big_tech_associations_openAI.csv"
    bigtech4 = "../results/cohere/big_tech/big_tech_associations_cohere.csv"
    bigtech5 = "../results/google/big_tech/big_tech_associations_google.csv"
    bigtech6 = "../results/microsoft/big_tech/big_tech_associations_microsoft.csv"
    bigtech7 = "../results/BGE/big_tech/big_tech_associations_BGE.csv"
    largestBigTechs.append(bigtech1)
    largestBigTechs.append(bigtech2)
    largestBigTechs.append(bigtech3)
    largestBigTechs.append(bigtech4)
    largestBigTechs.append(bigtech5)
    largestBigTechs.append(bigtech6)
    largestBigTechs.append(bigtech7)

    # Get gender stimuli
    female_stimuli = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
    male_stimuli = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']

    top_100k_embeddings = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    output_weats = "../results/glove/big_tech/test_glove_gender_big_tech_weats.csv"
    output_bigtechs = "../results/glove/big_tech/test_glove_big_tech_words.txt"
    pdf = "../plots/glove/bigtech/test_glove_gender_bigtech_ratio.pdf"
    process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf,female_stimuli, male_stimuli)

    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    # output_weats = "../results/fasttext/big_tech/ft_gender_big_tech_weats.csv"
    # output_bigtechs = "../results/fasttext/big_tech/ft_big_tech_words.txt"
    # pdf = "../plots/fasttext/bigtech/ft_gender_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf,female_stimuli, male_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    # output_weats = "../results/openAI/big_tech/openai_gender_big_tech_weats.csv"
    # output_bigtechs = "../results/openAI/big_tech/openai_big_tech_words.txt"
    # pdf = "../plots/openAI/bigtech/openai_gender_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf,female_stimuli, male_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    # output_weats = "../results/cohere/big_tech/cohere_gender_big_tech_weats.csv"
    # output_bigtechs = "../results/cohere/big_tech/cohere_big_tech_words.txt"
    # pdf = "../plots/cohere/bigtech/cohere_gender_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, female_stimuli, male_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    # output_weats = "../results/google/big_tech/google_gender_big_tech_weats.csv"
    # output_bigtechs = "../results/google/big_tech/google_big_tech_words.txt"
    # pdf = "../plots/google/bigtech/google_gender_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf,female_stimuli, male_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    # output_weats = "../results/microsoft/big_tech/microsoft_gender_big_tech_weats.csv"
    # output_bigtechs = "../results/microsoft/big_tech/microsoft_big_tech_words.txt"
    # pdf = "../plots/microsoft/bigtech/microsoft_gender_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf,female_stimuli, male_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    # output_weats = "../results/BGE/big_tech/BGE_gender_big_tech_weats.csv"
    # output_bigtechs = "../results/BGE/big_tech/BGE_big_tech_words.txt"
    # pdf = "../plots/BGE/bigtech/BGE_gender_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf,female_stimuli, male_stimuli)
    #
    # print("Finish gender class process")
    #
    # white_stimuli = ["caucasian", "white", "european", "american", "canadian", "australian", "british", "french",
    #                  "german", "italian"]
    # black_stimuli = ["black", "african", "african-american", "haitian", "jamaican", "nigerian", "kenyan", "ethiopian",
    #                  "egyptian", "congolese"]
    # asian_stimuli = ["asian", "brown", "chinese", "japanese", "korean", "indian", "filipino", "thai", "indonesian",
    #                  "pakistani"]
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    # output_weats = "../results/glove/big_tech/glove_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/glove/big_tech/glove_big_tech_words.txt"
    # pdf = "../plots/glove/bigtech/glove_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    # output_weats = "../results/fasttext/big_tech/ft_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/fasttext/big_tech/ft_big_tech_words.txt"
    # pdf = "../plots/fasttext/bigtech/ft_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    # output_weats = "../results/openAI/big_tech/openai_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/openAI/big_tech/openai_big_tech_words.txt"
    # pdf = "../plots/openAI/bigtech/openai_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    # output_weats = "../results/cohere/big_tech/cohere_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/cohere/big_tech/cohere_big_tech_words.txt"
    # pdf = "../plots/cohere/bigtech/cohere_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    # output_weats = "../results/google/big_tech/google_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/google/big_tech/google_big_tech_words.txt"
    # pdf = "../plots/google/bigtech/google_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    # output_weats = "../results/microsoft/big_tech/microsoft_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/microsoft/big_tech/microsoft_big_tech_words.txt"
    # pdf = "../plots/microsoft/bigtech/microsoft_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    # output_weats = "../results/BGE/big_tech/BGE_race_WB_big_tech_weats.csv"
    # output_bigtechs = "../results/BGE/big_tech/BGE_big_tech_words.txt"
    # pdf = "../plots/BGE/bigtech/BGE_race_WB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, black_stimuli)
    #
    # print("Finish first race class process")
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    # output_weats = "../results/glove/big_tech/glove_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/glove/big_tech/glove_big_tech_words.txt"
    # pdf = "../plots/glove/bigtech/glove_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    # output_weats = "../results/fasttext/big_tech/ft_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/fasttext/big_tech/ft_big_tech_words.txt"
    # pdf = "../plots/fasttext/bigtech/ft_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    # output_weats = "../results/openAI/big_tech/openai_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/openAI/big_tech/openai_big_tech_words.txt"
    # pdf = "../plots/openAI/bigtech/openai_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    # output_weats = "../results/cohere/big_tech/cohere_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/cohere/big_tech/cohere_big_tech_words.txt"
    # pdf = "../plots/cohere/bigtech/cohere_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    # output_weats = "../results/google/big_tech/google_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/google/big_tech/google_big_tech_words.txt"
    # pdf = "../plots/google/bigtech/google_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    # output_weats = "../results/microsoft/big_tech/microsoft_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/microsoft/big_tech/microsoft_big_tech_words.txt"
    # pdf = "../plots/microsoft/bigtech/microsoft_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    # output_weats = "../results/BGE/big_tech/BGE_race_WA_big_tech_weats.csv"
    # output_bigtechs = "../results/BGE/big_tech/BGE_big_tech_words.txt"
    # pdf = "../plots/BGE/bigtech/BGE_race_WA_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, white_stimuli, asian_stimuli)
    #
    # print("Finish second race class process")
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    # output_weats = "../results/glove/big_tech/glove_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/glove/big_tech/glove_big_tech_words.txt"
    # pdf = "../plots/glove/bigtech/glove_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    # output_weats = "../results/fasttext/big_tech/ft_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/fasttext/big_tech/ft_big_tech_words.txt"
    # pdf = "../plots/fasttext/bigtech/ft_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    # output_weats = "../results/openAI/big_tech/openai_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/openAI/big_tech/openai_big_tech_words.txt"
    # pdf = "../plots/openAI/bigtech/openai_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    # output_weats = "../results/cohere/big_tech/cohere_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/cohere/big_tech/cohere_big_tech_words.txt"
    # pdf = "../plots/cohere/bigtech/cohere_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    # output_weats = "../results/google/big_tech/google_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/google/big_tech/google_big_tech_words.txt"
    # pdf = "../plots/google/bigtech/google_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    # output_weats = "../results/microsoft/big_tech/microsoft_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/microsoft/big_tech/microsoft_big_tech_words.txt"
    # pdf = "../plots/microsoft/bigtech/microsoft_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # top_100k_embeddings = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    # output_weats = "../results/BGE/big_tech/BGE_race_AB_big_tech_weats.csv"
    # output_bigtechs = "../results/BGE/big_tech/BGE_big_tech_words.txt"
    # pdf = "../plots/BGE/bigtech/BGE_race_AB_bigtech_ratio.pdf"
    # process(top_100k_embeddings, largestBigTechs, output_weats, output_bigtechs, pdf, asian_stimuli, black_stimuli)
    #
    # print("Finish third race class process")