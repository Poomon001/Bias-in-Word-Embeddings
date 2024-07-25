'''
plot_cluster
Visualize word clusters derived from word embeddings. The word embeddings are reduced to two dimensions using the t-SNE
(t-Distributed Stochastic Neighbor Embedding) method to facilitate visualization.
'''

import pandas as pd
import matplotlib.pyplot as plt
import json

def plot(dat, pdf, clusterToTopic, title):
    with open(clusterToTopic, 'r') as json_file:
        cluster_data = json.load(json_file)

    data = pd.read_csv(dat, delim_whitespace=True)
    print(data.head())

    # Extract the data for plotting
    x = data['x']
    y = data['y']
    clusters = data['cluster']

    # Create a scatter plot
    plt.figure(figsize=(10, 8))

    # Scatter plot with different colors for different clusters
    scatter = plt.scatter(x, y, c=clusters, cmap='tab10', label=clusters)

    # Define the number-to-text mapping
    number_to_text = {int(key): value for key, value in cluster_data.items()}

    # Add a color bar
    cbar = plt.colorbar(scatter, label='Cluster')
    cbar.set_ticks(range(len(number_to_text)))
    cbar.set_ticklabels([number_to_text[i] for i in range(len(number_to_text))])

    # Add titles and labels
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()

    plt.plot()

    # Save the plot as a PDF
    plt.savefig(pdf, format='pdf')

    plt.show()

if __name__ == "__main__":
    femaleDat = "../results/glove/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/glove/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/glove/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/glove/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/glove/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/glove/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/fasttext/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/fasttext/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/fasttext/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/fasttext/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/fasttext/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/fasttext/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/openAI/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/openAI/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/openAI/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/openAI/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/openAI/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/openAI/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/cohere/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/cohere/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/cohere/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/cohere/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/cohere/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/cohere/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/google/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/google/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/google/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/google/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/google/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/google/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/microsoft/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/microsoft/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/microsoft/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/microsoft/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/microsoft/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/microsoft/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/BGE/clusters/tsne_clusters_female_over_male_vis_elkan_11.dat"
    maleDat = "../results/BGE/clusters/tsne_clusters_male_over_female_vis_elkan_11.dat"
    malePDF = "../plots/BGE/clusters/male_gender_clusters.pdf"
    femalePDF = "../plots/BGE/clusters/female_gender_clusters.pdf"
    maleClusterToTopic = "../results/BGE/clusters/male_over_female_clusters.json"
    femaleClusterToTopic = "../results/BGE/clusters/female_over_male_clusters.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    print("Finish gender class process")

    caucasianDat = "../results/glove/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/glove/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/glove/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/glove/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/glove/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/glove/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/fasttext/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/fasttext/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/fasttext/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/fasttext/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/fasttext/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/fasttext/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/openAI/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/openAI/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/openAI/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/openAI/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/openAI/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/openAI/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/cohere/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/cohere/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/cohere/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/cohere/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/cohere/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/cohere/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/google/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/google/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/google/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/google/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/google/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/google/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/microsoft/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/microsoft/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/microsoft/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/microsoft/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/microsoft/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/microsoft/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/BGE/clusters/tsne_clusters_caucasian_over_black_vis_elkan_11.dat"
    blackDat = "../results/BGE/clusters/tsne_clusters_black_over_caucasian_vis_elkan_11.dat"
    blackPDF = "../plots/BGE/clusters/black_race_WB_clusters.pdf"
    caucasianPDF = "../plots/BGE/clusters/caucasian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/BGE/clusters/black_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/BGE/clusters/caucasian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    print("Finish first race class process")

    caucasianDat = "../results/glove/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/glove/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/glove/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/glove/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/glove/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/glove/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/fasttext/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/fasttext/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/fasttext/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/fasttext/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/fasttext/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/fasttext/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/openAI/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/openAI/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/openAI/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/openAI/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/openAI/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/openAI/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/cohere/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/cohere/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/cohere/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/cohere/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/cohere/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/cohere/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/google/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/google/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/google/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/google/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/google/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/google/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/microsoft/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/microsoft/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/microsoft/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/microsoft/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/microsoft/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/microsoft/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    caucasianDat = "../results/BGE/clusters/tsne_clusters_caucasian_over_asian_vis_elkan_11.dat"
    asianDat = "../results/BGE/clusters/tsne_clusters_asian_over_caucasian_vis_elkan_11.dat"
    asianPDF = "../plots/BGE/clusters/asian_race_WB_clusters.pdf"
    caucasianPDF = "../plots/BGE/clusters/caucasian_race_WB_clusters.pdf"
    asianClusterToTopic = "../results/BGE/clusters/asian_over_caucasian_clusters.json"
    caucasianClusterToTopic = "../results/BGE/clusters/caucasian_over_asian_clusters.json"

    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")
    plot(caucasianDat, caucasianPDF, caucasianClusterToTopic, "Caucasian Scatter Plot of Words by Cluster")

    print("Finish second race class process")

    asianDat = "../results/glove/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/glove/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/glove/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/glove/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/glove/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/glove/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    asianDat = "../results/fasttext/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/fasttext/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/fasttext/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/fasttext/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/fasttext/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/fasttext/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    asianDat = "../results/openAI/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/openAI/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/openAI/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/openAI/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/openAI/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/openAI/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    asianDat = "../results/cohere/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/cohere/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/cohere/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/cohere/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/cohere/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/cohere/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    asianDat = "../results/google/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/google/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/google/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/google/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/google/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/google/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    asianDat = "../results/microsoft/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/microsoft/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/microsoft/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/microsoft/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/microsoft/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/microsoft/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    asianDat = "../results/BGE/clusters/tsne_clusters_asian_over_black_vis_elkan_11.dat"
    blackDat = "../results/BGE/clusters/tsne_clusters_black_over_asian_vis_elkan_11.dat"
    blackPDF = "../plots/BGE/clusters/black_race_WB_clusters.pdf"
    asianPDF = "../plots/BGE/clusters/asian_race_WB_clusters.pdf"
    blackClusterToTopic = "../results/BGE/clusters/black_over_asian_clusters.json"
    asianClusterToTopic = "../results/BGE/clusters/asian_over_black_clusters.json"

    plot(blackDat, blackPDF, blackClusterToTopic, "Black Scatter Plot of Words by Cluster")
    plot(asianDat, asianPDF, asianClusterToTopic, "Asian Scatter Plot of Words by Cluster")

    print("Finish third race class process")