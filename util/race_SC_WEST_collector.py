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

def process(embeddingInoutPath, resultOutputPath, tempDir, filename, first_stimuli, second_stimuli):
    # no idea what does it do
    embedding_df = pd.read_csv(embeddingInoutPath, sep=' ', header=None, index_col=0, na_values=None,
                               keep_default_na=False, encoding='utf-8')
    first_embeddings, second_embeddings = embedding_df.loc[first_stimuli].to_numpy(), embedding_df.loc[second_stimuli].to_numpy()
    embedding_targets = embedding_df.index.tolist()[:CAP]

    # Save embeddings to CSV files
    try:
        first_embeddings_df = pd.DataFrame(first_stimuli)
        first_embeddings_df.to_csv('../results/glove/most_frequency_words/first_embeddings.csv', index=False, header=False)

        second_embeddings_df = pd.DataFrame(second_embeddings)
        second_embeddings_df.to_csv('../results/glove/most_frequency_words/second_embeddings.csv', index=False, header=False)

        print("CSV files saved successfully.")
    except Exception as e:
        print("An error occurred while saving the CSV files:", e)

    # Non VAD WEAT
    # 10k WEATS at a time - 100k most frequent words - GloVe embedding
    for i in range(10):
        targets = embedding_targets[i * STEP:(i + 1) * STEP]
        bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(), first_embeddings, second_embeddings, PERMUTATIONS) for word in targets])
        bias_df = pd.DataFrame(bias_array, index=targets, columns=[f'{first_stimuli[0]}_effect_size', f'{first_stimuli[0]}_p_value'])
        bias_df.to_csv(path.join(tempDir, f'{filename}_100k_{i}.csv'), encoding='utf-8')

    print('Non-VAD')

    ''' TODO: Check if concat_ present any error in output dataset, comment this function and manually concat the csv if error presents '''
    # Concatenate and save 10k-word association dataframes
    concat_ = []
    for i in range(10):
        df = pd.read_csv(path.join(tempDir, f'{filename}_100k_{i}.csv'),
                         names=['word', f'{first_stimuli[0]}_effect_size', 'p_value'], skiprows=1, index_col='word', na_values=None,
                         keep_default_na=False, encoding='utf-8')
        concat_.append(df)

    full_df = pd.concat(concat_, axis=0)
    full_df.to_csv(resultOutputPath, encoding='utf-8')

    return bias_df

if __name__ == "__main__":
    white_stimuli = ["caucasian", "white", "european", "american", "canadian", "australian", "british", "french", "german", "italian"]
    black_stimuli = ["black", "african", "african-american", "haitian", "jamaican", "nigerian", "kenyan", "ethiopian", "egyptian", "congolese"]
    asian_stimuli = ["asian", "brown", "chinese", "japanese", "korean", "indian", "filipino", "thai", "indonesian", "pakistani"]

    tempDir = "../temp/glove"
    raw = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    filename = "glove_BW"
    result = "../results/glove/most_frequency_words/glove_race_WB_100000_most_frequency.csv"
    bias_glove_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "glove_WA"
    result = "../results/glove/most_frequency_words/glove_race_WA_100000_most_frequency.csv"
    bias_glove_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "glove_AB"
    result = "../results/glove/most_frequency_words/glove_race_AB_100000_most_frequency.csv"
    bias_glove_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)

    tempDir = "../temp/fasttext"
    raw = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    filename = "ft_BW"
    result = "../results/fasttext/most_frequency_words/ft_race_WB_100000_most_frequency.csv"
    bias_ft_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "ft_WA"
    result = "../results/fasttext/most_frequency_words/ft_race_WA_100000_most_frequency.csv"
    bias_glove_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "ft_AB"
    result = "../results/fasttext/most_frequency_words/ft_race_AB_100000_most_frequency.csv"
    bias_glove_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)

    tempDir = "../temp/openai"
    raw = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    result = "../results/openAI/most_frequency_words/openAI_race_WB_100000_most_frequency.csv"
    filename = "openai_BW"
    bias_openAI_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "openai_WA"
    result = "../results/openAI/most_frequency_words/openAI_race_WA_100000_most_frequency.csv"
    bias_openAI_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "openai_AB"
    result = "../results/openAI/most_frequency_words/openAI_race_AB_100000_most_frequency.csv"
    bias_openAI_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)

    tempDir = "../temp/cohere"
    raw = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    result = "../results/cohere/most_frequency_words/cohere_race_WB_100000_most_frequency.csv"
    filename = "cohere_BW"
    bias_cohere_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "cohere_WA"
    result = "../results/cohere/most_frequency_words/cohere_race_WA_100000_most_frequency.csv"
    bias_cohere_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "cohere_AB"
    result = "../results/cohere/most_frequency_words/cohere_race_AB_100000_most_frequency.csv"
    bias_cohere_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)

    tempDir = "../temp/google"
    raw = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    result = "../results/google/most_frequency_words/google_race_WB_100000_most_frequency.csv"
    filename = "google_BW"
    bias_google_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "google_WA"
    result = "../results/google/most_frequency_words/google_race_WA_100000_most_frequency.csv"
    bias_google_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "google_AB"
    result = "../results/google/most_frequency_words/google_race_AB_100000_most_frequency.csv"
    bias_google_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)

    tempDir = "../temp/microsoft"
    raw = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    result = "../results/microsoft/most_frequency_words/microsoft_race_WB_100000_most_frequency.csv"
    filename = "microsoft_BW"
    bias_microsoft_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "microsoft_WA"
    result = "../results/microsoft/most_frequency_words/microsoft_race_WA_100000_most_frequency.csv"
    bias_microsoft_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "microsoft_AB"
    result = "../results/microsoft/most_frequency_words/microsoft_race_AB_100000_most_frequency.csv"
    bias_microsoft_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)

    tempDir = "../temp/BGE"
    raw = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    result = "../results/BGE/most_frequency_words/BGE_race_WB_100000_most_frequency.csv"
    filename = "BGE_BW"
    bias_BGE_100000 = process(raw, result, tempDir, filename, white_stimuli, black_stimuli)
    filename = "BGE_WA"
    result = "../results/BGE/most_frequency_words/BGE_race_WA_100000_most_frequency.csv"
    bias_BGE_100000 = process(raw, result, tempDir, filename, white_stimuli, asian_stimuli)
    filename = "BGE_AB"
    result = "../results/BGE/most_frequency_words/BGE_race_AB_100000_most_frequency.csv"
    bias_BGE_100000 = process(raw, result, tempDir, filename, asian_stimuli, black_stimuli)
