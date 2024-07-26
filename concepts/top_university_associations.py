import numpy as np
import pandas as pd
import csv

def process(top_100k_embeddings, top_50_university_embeddings, result):
    top_100k_embedding_df = pd.read_csv(top_100k_embeddings, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
    top_50_embedding_df = pd.read_csv(top_50_university_embeddings, sep=',', header=0, index_col=0)
    top_50_embedding_df.columns = top_100k_embedding_df.columns

    embedding_df = pd.concat([top_100k_embedding_df, top_50_embedding_df])

    # Get mean cosine similarities with Big Tech words

    top_university_words = []

    # Open and read the file
    with open("top_50_universities.txt", "r", encoding='utf-8') as file:
        for line in file:
            cleaned_line = line.strip()
            if cleaned_line:
                top_university_words.append(cleaned_line)

    top_university_embs = embedding_df.loc[[word for word in top_university_words if word in embedding_df.index]].to_numpy()
    top_university_normed = top_university_embs / np.linalg.norm(top_university_embs, axis=-1, keepdims=True)

    all_embs = embedding_df.to_numpy()
    all_embs_normed = all_embs / np.linalg.norm(all_embs, axis=-1, keepdims=True)

    associations = all_embs_normed @ top_university_normed.T
    means = np.mean(associations, axis=1)

    # Write dataframe to file

    big_tech_df = pd.DataFrame(means, index=embedding_df.index.tolist(), columns=['top_university_es'])
    largest = big_tech_df.nlargest(10000, 'top_university_es')

    largest.to_csv(result)

if __name__ == "__main__":
    process("D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt",
            "D:/Honour_Thesis_Data/BGE/BGE_top_50_universities.csv",
            "../results/BGE/top_university/top_university_associations_BGE.csv")

    process("D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt",
            "D:/Honour_Thesis_Data/cohere/cohere_top_50_universities.csv",
            "../results/cohere/top_university/top_university_associations_cohere.csv")