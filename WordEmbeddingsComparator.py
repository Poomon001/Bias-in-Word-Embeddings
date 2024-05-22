import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

def plot(bias_df, title, output):
    bias_df['female_effect_size'].plot(kind='bar')

    # Adding labels and title
    plt.xlabel('VAD Words')
    plt.ylabel('Female Effect Size')
    plt.title(f'Female Effect Size by VAD Words: {title}')
    plt.savefig(output)
    plt.show()

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
female_stimuli = ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]
male_stimuli = ["male", "he", "him", "his", "man", "boy", "son", "brother"]

def process(embeddingInoutPath, resultOutputPath):
    # no idea what does it do
    embedding_df = pd.read_csv(embeddingInoutPath, sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False)
    female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()
    embedding_targets = embedding_df.index.tolist()[:100]

    # Save embeddings to CSV files
    try:
        female_embeddings_df = pd.DataFrame(female_embeddings)
        female_embeddings_df.to_csv('results/female_OpenAI_embeddings.csv', index=False, header=False)

        male_embeddings_df = pd.DataFrame(male_embeddings)
        male_embeddings_df.to_csv('results/male_OpenAI_embeddings.csv', index=False, header=False)

        print("CSV files saved successfully.")
    except Exception as e:
        print("An error occurred while saving the CSV files:", e)

    #NRC-VAD Dataframe
    vad_df = pd.read_table(f'raw/NRC-VAD-Lexicon.txt',sep='\t',index_col=0, na_values=None, keep_default_na=False)
    vad_words = vad_df.index.tolist()
    vad_words = [word for word in vad_words if word in embedding_df.index]

    print("VAD word exclusion:", set(embedding_targets) - set(vad_words))

    gender_biases, p_values = [],[]

    #VAD WEATs - GloVe embedding
    bias_array = [SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in vad_words]
    bias_df = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
    bias_df.reset_index(inplace=True)
    bias_df.rename(columns={'index': 'word'}, inplace=True)
    bias_df.to_csv(resultOutputPath, index=False)
    print('GloVe VAD')
    return bias_df


if __name__ == "__main__":
    bias_df = process("openAI/openAI_100.txt", "results/openAI_vad_words.csv")
    plot(bias_df, "OpenAI", "plots/OpenAI_effect_size_plot.png")

    bias_df = process("raw/glove_embeddings_100.txt", "results/glove_vad_words.csv")
    plot(bias_df, "Glove", "plots/glove_effect_size_plot.png")

    bias_df = process("raw/ft_embeddings_100.txt", "results/ft_vad_words.csv")
    plot(bias_df, "FastText", "plots/ft_effect_size_plot.png")