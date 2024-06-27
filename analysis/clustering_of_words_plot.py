import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
import random
from os import path
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def plot(embedding_100k_file, top_100k_file, dir, dirPDF):
    # Read in gender association file and get most associated female and male words
    embedding_100k = pd.read_csv(top_100k_file, na_values=None, keep_default_na=False,names=['word','female_effect_size','p_value'],skiprows=1)

    ''' check word: 1114 and 80009 if there is any issues '''
    embedding_female = embedding_100k.loc[
        (embedding_100k['female_effect_size'] >= 0.5) &
        (embedding_100k['p_value'] <= 0.05)
        ]
    embedding_top_female = embedding_female.head(1000)
    top_female_words = embedding_top_female['word'].tolist()

    embedding_male = embedding_100k.loc[
        (embedding_100k['female_effect_size'] <= -.5) &
        (embedding_100k['p_value'] >= .95)
        ]
    embedding_top_male = embedding_male.head(1000)
    top_male_words = embedding_top_male['word'].tolist()

    print(top_female_words)
    print(top_male_words)

    embedding_df = pd.read_csv(embedding_100k_file, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=100000)

    print(embedding_df)

    embeddings_female = embedding_df.loc[top_female_words]
    embeddings_male = embedding_df.loc[top_male_words]

    target_data_female = embeddings_female.to_numpy()
    target_data_male = embeddings_male.to_numpy()

    # Use elbow method to assess stopping point for female and male clusters

    INIT = 3
    ITERS = 26

    wcss = []

    for i in range(INIT, ITERS):
        kmeans = KMeans(n_clusters=i, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000, n_init=100)
        kmeans.fit(target_data_female)
        wcss.append(kmeans.inertia_)

    print(wcss)

    plt.plot([i for i in range(INIT, ITERS)], wcss)
    plt.xticks([i for i in range(INIT, ITERS)])
    plt.savefig(path.join(dirPDF, f'female_kmean.pdf'), format='pdf')

    wcss = []

    for i in range(INIT, ITERS):
        kmeans = KMeans(n_clusters=i, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000, n_init=100)
        kmeans.fit(target_data_male)
        wcss.append(kmeans.inertia_)

    print(wcss)

    plt.plot([i for i in range(INIT, ITERS)], wcss)
    plt.xticks([i for i in range(INIT, ITERS)])
    plt.savefig(path.join(dirPDF, f'male_kmean.pdf'), format='pdf')

    print("Complete kmean")

    # K-Means clustering and transformed coordinates
    NUM_CLUSTERS = 11
    kmeans_female = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000,
                           n_init=100).fit(target_data_female)
    kmeans_female_transform = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++',
                                     max_iter=1000, n_init=100).fit_transform(target_data_female)

    kmeans_male = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000,
                         n_init=100).fit(target_data_male)
    kmeans_male_transform = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++',
                                   max_iter=1000, n_init=100).fit_transform(target_data_male)

    # T-SNE coordinates
    reduced_dims_female = TSNE().fit_transform(kmeans_female_transform.squeeze())
    tsne_df_female = pd.DataFrame(reduced_dims_female, index=top_female_words, columns=['x', 'y'])
    tsne_df_female['word'] = top_female_words
    tsne_df_female['cluster'] = kmeans_female.labels_
    tsne_df_female.to_csv(path.join(dir, f'tsne_clusters_female_1k_{NUM_CLUSTERS}.csv'))

    tsne_female_x = tsne_df_female['x'].tolist()
    tsne_female_y = tsne_df_female['y'].tolist()

    reduced_dims_male = TSNE().fit_transform(kmeans_male_transform.squeeze())
    tsne_df_male = pd.DataFrame(reduced_dims_male, index=top_male_words, columns=['x', 'y'])
    tsne_df_male['word'] = top_male_words
    tsne_df_male['cluster'] = kmeans_male.labels_
    tsne_df_male.to_csv(path.join(dir, f'tsne_clusters_male_1k_{NUM_CLUSTERS}.csv'))

    tsne_male_x = tsne_df_male['x'].tolist()
    tsne_male_y = tsne_df_male['y'].tolist()

    # Write cluster coordinates to .dat file
    write_string = 'x\ty\tcluster\tword\n' + '\n'.join(['\t'.join(
        [str(tsne_female_x[i]), str(tsne_female_y[i]), str(kmeans_female.labels_[i]), str(top_female_words[i])]) for i
                                                        in range(len(tsne_female_x))])
    with open(path.join(dir, f'tsne_clusters_female_male_vis_elkan_{NUM_CLUSTERS}.dat'), 'w', encoding='utf8') as writer:
        writer.write(write_string)

    write_string = 'x\ty\tcluster\tword\n' + '\n'.join(
        ['\t'.join([str(tsne_male_x[i]), str(tsne_male_y[i]), str(kmeans_male.labels_[i]), str(top_male_words[i])]) for
         i in range(len(tsne_male_x))])
    with open(path.join(dir, f'tsne_clusters_male_male_vis_elkan_{NUM_CLUSTERS}.dat'), 'w', encoding='utf8') as writer:
        writer.write(write_string)

    female_write_string, male_write_string = '', ''

    # Write female and male clustered words to text files
    for i in range(NUM_CLUSTERS):
        cluster_df_female = tsne_df_female.loc[tsne_df_female['cluster'] == i]
        cluster_df_male = tsne_df_male.loc[tsne_df_male['cluster'] == i]

        female_write_string += f'Cluster {i}:' + ', '.join(
            sorted(cluster_df_female.index.tolist(), key=str.lower)) + '\n'
        male_write_string += f'Cluster {i}:' + ', '.join(sorted(cluster_df_male.index.tolist(), key=str.lower)) + '\n'

    with open(path.join(dir, f'female_clusters_{NUM_CLUSTERS}.txt'), 'w', encoding='utf8') as writer:
        writer.write(female_write_string)

    with open(path.join(dir, f'male_clusters_{NUM_CLUSTERS}.txt'), 'w', encoding='utf8') as writer:
        writer.write(male_write_string)


if __name__ == "__main__":
    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_100k_file = "../results/six_methods/most_frequency_words/glove_100000_most_frequency.csv"
    dir = "../results/six_methods/clusters"
    dirPDF = "../plots/six_methods/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_100k_file = "../results/openAI/most_frequency_words/openAI_100000_most_frequency.csv"
    dir = "../results/openAI/clusters"
    dirPDF = "../plots/openAI/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_100k_file = "../results/fasttext/most_frequency_words/ft_100000_most_frequency.csv"
    dir = "../results/fasttext/clusters"
    dirPDF = "../plots/fasttext/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_100k_file = "../results/cohere/most_frequency_words/cohere_100000_most_frequency.csv"
    dir = "../results/cohere/clusters"
    dirPDF = "../plots/cohere/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_100k_file = "../results/google/most_frequency_words/google_100000_most_frequency.csv"
    dir = "../results/google/clusters"
    dirPDF = "../plots/google/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_100k_file = "../results/microsoft/most_frequency_words/microsoft_100000_most_frequency.csv"
    dir = "../results/microsoft/clusters"
    dirPDF = "../plots/microsoft/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft_norm/microsoft_norm_100000_most_freq_skip.txt"
    top_100k_file = "../results/microsoft_norm/most_frequency_words/microsoft_norm_100000_most_frequency.csv"
    dir = "../results/microsoft_norm/clusters"
    dirPDF = "../plots/microsoft_norm/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_100k_file = "../results/BGE/most_frequency_words/BGE_100000_most_frequency.csv"
    dir = "../results/BGE/clusters"
    dirPDF = "../plots/BGE/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF)