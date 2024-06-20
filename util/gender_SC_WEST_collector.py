import numpy as np
from scipy.stats import norm
import pandas as pd
from os import path

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

PERMUTATIONS = 10000
CAP = 100000
STEP = 10000
female_stimuli = ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]
male_stimuli = ["male", "he", "him", "his", "man", "boy", "son", "brother"]

def process(embeddingInoutPath, resultOutputPath, tempDir, filename):
    # no idea what does it do
    embedding_df = pd.read_csv(embeddingInoutPath, sep=' ', header=None, index_col=0, na_values=None,
                               keep_default_na=False)
    female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()
    embedding_targets = embedding_df.index.tolist()[:CAP]

    # Save embeddings to CSV files
    try:
        female_embeddings_df = pd.DataFrame(female_embeddings)
        female_embeddings_df.to_csv('../results/six_methods/most_frequency_words/female_embeddings.csv', index=False, header=False)

        male_embeddings_df = pd.DataFrame(male_embeddings)
        male_embeddings_df.to_csv('../results/six_methods/most_frequency_words/male__embeddings.csv', index=False, header=False)

        print("CSV files saved successfully.")
    except Exception as e:
        print("An error occurred while saving the CSV files:", e)

    # Non VAD WEAT
    # 10k WEATS at a time - 100k most frequent words - GloVe embedding
    for i in range(10):
        targets = embedding_targets[i * STEP:(i + 1) * STEP]
        bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(), female_embeddings, male_embeddings, PERMUTATIONS) for word in targets])
        bias_df = pd.DataFrame(bias_array, index=targets, columns=['female_effect_size', 'female_p_value'])
        bias_df.to_csv(path.join(tempDir, f'{filename}_100k_{i}.csv'))

    print('Non-VAD')

    # Concatenate and save 10k-word association dataframes
    concat_ = []
    for i in range(10):
        df = pd.read_csv(path.join(tempDir, f'{filename}_100k_{i}.csv'),
                         names=['word', 'female_effect_size', 'p_value'], skiprows=1, index_col='word', na_values=None,
                         keep_default_na=False)
        concat_.append(df)

    full_df = pd.concat(concat_, axis=0)
    full_df.to_csv(resultOutputPath)

    return bias_df

if __name__ == "__main__":
    tempDir = "../temp/glove"
    raw = "../raw/glove_100000_most_freq_skip.txt"
    result = "../results/six_methods/most_frequency_words/glove_100000_most_frequency.csv"
    filename = "glove"
    # bias_glove_100000 = process(raw, result, tempDir, filename)

    tempDir = "../temp/openai"
    raw = "../openAI/openAI_100000_skip.txt"
    result = "../results/openAI/most_frequency_words/openAI_100000_most_frequency.csv"
    filename = "openai"
    # bias_openAI_100000 = process(raw, result, tempDir, filename)

    tempDir = "../temp/fasttext"
    raw = "../raw/ft_100000_most_freq_skip.csv"
    result = "../results/fasttext/most_frequency_words/ft_100000_most_frequency.csv"
    tempDir = "../temp/fasttext"
    filename = "ft"
    # bias_ft_100000 = process(raw, result, tempDir, filename)

    tempDir = "../temp/cohere"
    raw = "../cohere/cohere_100000_most_freq_skip.txt"
    result = "../results/cohere/most_frequency_words/cohere_100000_most_frequency.csv"
    filename = "cohere"
    # bias_cohere_100000 = process(raw, result, tempDir, filename)

    tempDir = "../temp/google"
    raw = "../google/google_100000_most_freq_skip.txt"
    result = "../results/google/most_frequency_words/google_100000_most_frequency.csv"
    filename = "cohere"
    bias_google_100000 = process(raw, result, tempDir, filename)