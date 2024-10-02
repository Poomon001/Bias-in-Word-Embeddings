'''
plot_cluster
Visualize word clusters derived from word embeddings. The word embeddings are reduced to two dimensions using the t-SNE
(t-Distributed Stochastic Neighbor Embedding) method to facilitate visualization.
'''

import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.colors import ListedColormap


def plot(dat, pdf, clusterToTopic, title):
    with open(clusterToTopic, 'r') as json_file:
        cluster_data = json.load(json_file)
    data = pd.read_csv(dat, delim_whitespace=True)

    x = data['x']
    y = data['y']

    # Define colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#FFD700']
    colors = colors[:len(cluster_data)]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = plt.scatter(x, y, c=data['cluster'], cmap=ListedColormap(colors))

    # Create legend
    color_patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
    legend_labels = list(cluster_data.values())

    # Place legend
    ax.legend(color_patches, legend_labels,
              loc='center left',
              bbox_to_anchor=(1.05, 0.5),
              title="Clusters",
              title_fontsize=12,
              fontsize=10,
              handleheight=2.0,
              labelspacing=1.0)

    # Adjust layout
    plt.tight_layout()

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Save and show plot
    plt.savefig(pdf, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()

if __name__ == "__main__":
    femaleDat = "../results/glove/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/glove/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/glove/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/glove/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/glove/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/glove/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - GloVe")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - GloVe")

    femaleDat = "../results/fasttext/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/fasttext/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/fasttext/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/fasttext/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/fasttext/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/fasttext/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - FastText")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - FastText")

    femaleDat = "../results/openAI/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/openAI/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/openAI/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/openAI/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/openAI/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/openAI/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - OpenAI")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - OpenAI")

    femaleDat = "../results/cohere/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/cohere/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/cohere/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/cohere/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/cohere/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/cohere/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - Cohere")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - Cohere")

    femaleDat = "../results/google/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/google/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/google/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/google/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/google/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/google/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - Google")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - Google")

    femaleDat = "../results/microsoft/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/microsoft/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/microsoft/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/microsoft/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/microsoft/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/microsoft/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - Microsoft")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - Microsoft")

    femaleDat = "../results/BGE/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/BGE/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/BGE/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/BGE/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/BGE/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/BGE/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster - BGE")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster - BGE")

    print("Finish gender class process")

    caucasianDat = "../results/glove/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/glove/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/glove/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/glove/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/glove/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/glove/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - GloVe")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - GloVe")

    caucasianDat = "../results/fasttext/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/fasttext/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/fasttext/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/fasttext/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/fasttext/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/fasttext/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - FastText")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - FastText")

    caucasianDat = "../results/openAI/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/openAI/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/openAI/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/openAI/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/openAI/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/openAI/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - OpenAI")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - OpenAI")

    caucasianDat = "../results/cohere/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/cohere/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/cohere/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/cohere/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/cohere/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/cohere/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - Cohere")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - Cohere")

    caucasianDat = "../results/google/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/google/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/google/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/google/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/google/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/google/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - Google")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - Google")

    caucasianDat = "../results/microsoft/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/microsoft/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/microsoft/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/microsoft/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/microsoft/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/microsoft/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - Microsoft")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - Microsoft")

    caucasianDat = "../results/BGE/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/BGE/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/BGE/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/BGE/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/BGE/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/BGE/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - BGE")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - BGE")

    print("Finish first race class process")

    caucasianDat = "../results/glove/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/glove/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/glove/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/glove/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/glove/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/glove/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - GloVe")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - GloVe")

    caucasianDat = "../results/fasttext/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/fasttext/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/fasttext/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/fasttext/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/fasttext/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/fasttext/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - FastText")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - FastText")

    caucasianDat = "../results/openAI/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/openAI/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/openAI/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/openAI/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/openAI/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/openAI/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - OpenAI")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - OpenAI")

    caucasianDat = "../results/cohere/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/cohere/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/cohere/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/cohere/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/cohere/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/cohere/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - Cohere")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - Cohere")

    caucasianDat = "../results/google/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/google/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/google/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/google/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/google/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/google/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - Google")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - Google")

    caucasianDat = "../results/microsoft/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/microsoft/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/microsoft/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/microsoft/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/microsoft/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/microsoft/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - Microsoft")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - Microsoft")

    caucasianDat = "../results/BGE/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/BGE/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/BGE/clusters/asian_race_WA_clusters.pdf"
    caucasianPDF = "../plots/BGE/clusters/caucasian_race_WA_clusters.pdf"
    asianClusterToTopic = "../results/BGE/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/BGE/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - BGE")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster - BGE")

    print("Finish second race class process")

    asianDat = "../results/glove/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/glove/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/glove/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/glove/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/glove/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/glove/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - GloVe")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - GloVe")

    asianDat = "../results/fasttext/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/fasttext/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/fasttext/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/fasttext/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/fasttext/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/fasttext/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - FastText")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - FastText")

    asianDat = "../results/openAI/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/openAI/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/openAI/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/openAI/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/openAI/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/openAI/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - OpenAI")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - OpenAI")

    asianDat = "../results/cohere/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/cohere/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/cohere/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/cohere/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/cohere/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/cohere/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - Cohere")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - Cohere")

    asianDat = "../results/google/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/google/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/google/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/google/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/google/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/google/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - Google")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - Google")

    asianDat = "../results/microsoft/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/microsoft/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/microsoft/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/microsoft/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/microsoft/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/microsoft/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - Microsoft")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - Microsoft")

    asianDat = "../results/BGE/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/BGE/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/BGE/clusters/black_race_AB_clusters.pdf"
    asianPDF = "../plots/BGE/clusters/asian_race_AB_clusters.pdf"
    blackClusterToTopic = "../results/BGE/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/BGE/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster - BGE")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster - BGE")

    print("Finish third race class process")