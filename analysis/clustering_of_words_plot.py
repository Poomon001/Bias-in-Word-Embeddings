import pandas as pd
from os import path
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups):
    # Read in association file and get most associated group1 and group2 words
    embedding_100k = pd.read_csv(top_100k_file, na_values=None, keep_default_na=False,names=['word',f'{groups[0]}_effect_size','p_value'],skiprows=1)

    ''' check word: 1114 and 80009 if there is any issues '''
    embedding_group1 = embedding_100k.loc[
        (embedding_100k[f'{groups[0]}_effect_size'] >= 0.5) &
        (embedding_100k['p_value'] <= 0.05)
        ]
    embedding_top_group1 = embedding_group1.head(1000)
    top_group1_words = embedding_top_group1['word'].tolist()

    embedding_group2 = embedding_100k.loc[
        (embedding_100k[f'{groups[0]}_effect_size'] <= -.5) &
        (embedding_100k['p_value'] >= .95)
        ]
    embedding_top_group2 = embedding_group2.head(1000)
    top_group2_words = embedding_top_group2['word'].tolist()

    print(top_group1_words)
    print(top_group2_words)

    embedding_df = pd.read_csv(embedding_100k_file, sep=' ', header=None, index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=100000)

    print(embedding_df)

    embeddings_group1 = embedding_df.loc[top_group1_words]
    embeddings_group2 = embedding_df.loc[top_group2_words]

    target_data_group1 = embeddings_group1.to_numpy()
    target_data_group2 = embeddings_group2.to_numpy()

    # Use elbow method to assess stopping point for group1 and group2 clusters

    INIT = 3
    ITERS = 26

    wcss = []

    for i in range(INIT, ITERS):
        kmeans = KMeans(n_clusters=i, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000, n_init=100)
        kmeans.fit(target_data_group1)
        wcss.append(kmeans.inertia_)

    print(wcss)

    plt.plot([i for i in range(INIT, ITERS)], wcss)
    plt.xticks([i for i in range(INIT, ITERS)])
    plt.savefig(path.join(dirPDF, f'{groups[0]}_kmean.pdf'), format='pdf')

    wcss = []

    for i in range(INIT, ITERS):
        kmeans = KMeans(n_clusters=i, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000, n_init=100)
        kmeans.fit(target_data_group2)
        wcss.append(kmeans.inertia_)

    print(wcss)

    plt.plot([i for i in range(INIT, ITERS)], wcss)
    plt.xticks([i for i in range(INIT, ITERS)])
    plt.savefig(path.join(dirPDF, f'{groups[1]}_kmean.pdf'), format='pdf')

    print("Complete kmean")

    # K-Means clustering and transformed coordinates
    NUM_CLUSTERS = 11
    kmeans_group1 = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000,
                           n_init=100).fit(target_data_group1)
    kmeans_group1_transform = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++',
                                     max_iter=1000, n_init=100).fit_transform(target_data_group1)

    kmeans_group2 = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++', max_iter=1000,
                         n_init=100).fit(target_data_group2)
    kmeans_group2_transform = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='elkan', init='k-means++',
                                   max_iter=1000, n_init=100).fit_transform(target_data_group2)

    # T-SNE coordinates
    reduced_dims_group1 = TSNE().fit_transform(kmeans_group1_transform.squeeze())
    tsne_df_group1 = pd.DataFrame(reduced_dims_group1, index=top_group1_words, columns=['x', 'y'])
    tsne_df_group1['word'] = top_group1_words
    tsne_df_group1['cluster'] = kmeans_group1.labels_
    tsne_df_group1.to_csv(path.join(dir, f'tsne_clusters_{groups[0]}_over_{groups[1]}_1k_{NUM_CLUSTERS}.csv'))

    tsne_group1_x = tsne_df_group1['x'].tolist()
    tsne_group1_y = tsne_df_group1['y'].tolist()

    reduced_dims_group2 = TSNE().fit_transform(kmeans_group2_transform.squeeze())
    tsne_df_group2 = pd.DataFrame(reduced_dims_group2, index=top_group2_words, columns=['x', 'y'])
    tsne_df_group2['word'] = top_group2_words
    tsne_df_group2['cluster'] = kmeans_group2.labels_
    tsne_df_group2.to_csv(path.join(dir, f'tsne_clusters_{groups[1]}_over_{groups[0]}_1k_{NUM_CLUSTERS}.csv'))

    tsne_group2_x = tsne_df_group2['x'].tolist()
    tsne_group2_y = tsne_df_group2['y'].tolist()

    # Write cluster coordinates to .dat file
    write_string = 'x\ty\tcluster\tword\n' + '\n'.join(['\t'.join(
        [str(tsne_group1_x[i]), str(tsne_group1_y[i]), str(kmeans_group1.labels_[i]), str(top_group1_words[i])]) for i
                                                        in range(len(tsne_group1_x))])
    with open(path.join(dir, f'tsne_clusters_{groups[0]}_over_{groups[1]}_vis_elkan_{NUM_CLUSTERS}.dat'), 'w', encoding='utf8') as writer:
        writer.write(write_string)

    write_string = 'x\ty\tcluster\tword\n' + '\n'.join(
        ['\t'.join([str(tsne_group2_x[i]), str(tsne_group2_y[i]), str(kmeans_group2.labels_[i]), str(top_group2_words[i])]) for
         i in range(len(tsne_group2_x))])
    with open(path.join(dir, f'tsne_clusters_{groups[1]}_over_{groups[0]}_vis_elkan_{NUM_CLUSTERS}.dat'), 'w', encoding='utf8') as writer:
        writer.write(write_string)

    group1_write_string, group2_write_string = '', ''

    # Write group1 and group2 clustered words to text files
    for i in range(NUM_CLUSTERS):
        cluster_df_group1 = tsne_df_group1.loc[tsne_df_group1['cluster'] == i]
        cluster_df_group2 = tsne_df_group2.loc[tsne_df_group2['cluster'] == i]

        group1_write_string += f'Cluster {i}:' + ', '.join(
            sorted(cluster_df_group1.index.tolist(), key=str.lower)) + '\n'
        group2_write_string += f'Cluster {i}:' + ', '.join(sorted(cluster_df_group2.index.tolist(), key=str.lower)) + '\n'

    with open(path.join(dir, f'{groups[0]}_over_{groups[1]}_clusters_{NUM_CLUSTERS}.txt'), 'w', encoding='utf8') as writer:
        writer.write(group1_write_string)

    with open(path.join(dir, f'{groups[1]}_over_{groups[0]}_clusters_{NUM_CLUSTERS}.txt'), 'w', encoding='utf8') as writer:
        writer.write(group2_write_string)


if __name__ == "__main__":
    groups = ["female", "male"]

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_100k_file = "../results/glove/most_frequency_words/glove_gender_100000_most_frequency.csv"
    dir = "../results/glove/clusters"
    dirPDF = "../plots/glove/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_100k_file = "../results/openAI/most_frequency_words/openAI_gender_100000_most_frequency.csv"
    dir = "../results/openAI/clusters"
    dirPDF = "../plots/openAI/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_100k_file = "../results/fasttext/most_frequency_words/ft_gender_100000_most_frequency.csv"
    dir = "../results/fasttext/clusters"
    dirPDF = "../plots/fasttext/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_100k_file = "../results/cohere/most_frequency_words/cohere_gender_100000_most_frequency.csv"
    dir = "../results/cohere/clusters"
    dirPDF = "../plots/cohere/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_100k_file = "../results/google/most_frequency_words/google_gender_100000_most_frequency.csv"
    dir = "../results/google/clusters"
    dirPDF = "../plots/google/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_100k_file = "../results/microsoft/most_frequency_words/microsoft_gender_100000_most_frequency.csv"
    dir = "../results/microsoft/clusters"
    dirPDF = "../plots/microsoft/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_100k_file = "../results/BGE/most_frequency_words/BGE_gender_100000_most_frequency.csv"
    dir = "../results/BGE/clusters"
    dirPDF = "../plots/BGE/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    print("Finish gender class process")

    groups = ["caucasian", "black"]

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_100k_file = "../results/glove/most_frequency_words/glove_race_WB_100000_most_frequency.csv"
    dir = "../results/glove/clusters"
    dirPDF = "../plots/glove/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_100k_file = "../results/openAI/most_frequency_words/openAI_race_WB_100000_most_frequency.csv"
    dir = "../results/openAI/clusters"
    dirPDF = "../plots/openAI/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_100k_file = "../results/fasttext/most_frequency_words/ft_race_WB_100000_most_frequency.csv"
    dir = "../results/fasttext/clusters"
    dirPDF = "../plots/fasttext/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_100k_file = "../results/cohere/most_frequency_words/cohere_race_WB_100000_most_frequency.csv"
    dir = "../results/cohere/clusters"
    dirPDF = "../plots/cohere/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_100k_file = "../results/google/most_frequency_words/google_race_WB_100000_most_frequency.csv"
    dir = "../results/google/clusters"
    dirPDF = "../plots/google/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_100k_file = "../results/microsoft/most_frequency_words/microsoft_race_WB_100000_most_frequency.csv"
    dir = "../results/microsoft/clusters"
    dirPDF = "../plots/microsoft/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_100k_file = "../results/BGE/most_frequency_words/BGE_race_WB_100000_most_frequency.csv"
    dir = "../results/BGE/clusters"
    dirPDF = "../plots/BGE/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    print("Finish first race class process")

    groups = ["caucasian", "asian"]

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_100k_file = "../results/glove/most_frequency_words/glove_race_WA_100000_most_frequency.csv"
    dir = "../results/glove/clusters"
    dirPDF = "../plots/glove/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_100k_file = "../results/openAI/most_frequency_words/openAI_race_WA_100000_most_frequency.csv"
    dir = "../results/openAI/clusters"
    dirPDF = "../plots/openAI/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_100k_file = "../results/fasttext/most_frequency_words/ft_race_WA_100000_most_frequency.csv"
    dir = "../results/fasttext/clusters"
    dirPDF = "../plots/fasttext/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_100k_file = "../results/cohere/most_frequency_words/cohere_race_WA_100000_most_frequency.csv"
    dir = "../results/cohere/clusters"
    dirPDF = "../plots/cohere/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_100k_file = "../results/google/most_frequency_words/google_race_WA_100000_most_frequency.csv"
    dir = "../results/google/clusters"
    dirPDF = "../plots/google/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_100k_file = "../results/microsoft/most_frequency_words/microsoft_race_WA_100000_most_frequency.csv"
    dir = "../results/microsoft/clusters"
    dirPDF = "../plots/microsoft/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_100k_file = "../results/BGE/most_frequency_words/BGE_race_WA_100000_most_frequency.csv"
    dir = "../results/BGE/clusters"
    dirPDF = "../plots/BGE/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    print("Finish second race class process")

    groups = ["asian", "black"]

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_100k_file = "../results/glove/most_frequency_words/glove_race_AB_100000_most_frequency.csv"
    dir = "../results/glove/clusters"
    dirPDF = "../plots/glove/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_100k_file = "../results/openAI/most_frequency_words/openAI_race_AB_100000_most_frequency.csv"
    dir = "../results/openAI/clusters"
    dirPDF = "../plots/openAI/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_100k_file = "../results/fasttext/most_frequency_words/ft_race_AB_100000_most_frequency.csv"
    dir = "../results/fasttext/clusters"
    dirPDF = "../plots/fasttext/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_100k_file = "../results/cohere/most_frequency_words/cohere_race_AB_100000_most_frequency.csv"
    dir = "../results/cohere/clusters"
    dirPDF = "../plots/cohere/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_100k_file = "../results/google/most_frequency_words/google_race_AB_100000_most_frequency.csv"
    dir = "../results/google/clusters"
    dirPDF = "../plots/google/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_100k_file = "../results/microsoft/most_frequency_words/microsoft_race_AB_100000_most_frequency.csv"
    dir = "../results/microsoft/clusters"
    dirPDF = "../plots/microsoft/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_100k_file = "../results/BGE/most_frequency_words/BGE_race_AB_100000_most_frequency.csv"
    dir = "../results/BGE/clusters"
    dirPDF = "../plots/BGE/clusters"
    plot(embedding_100k_file, top_100k_file, dir, dirPDF, groups)

    print("Finish third race class process")