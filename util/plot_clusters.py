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

    # Save the plot as a PDF
    plt.savefig(pdf, format='pdf')

    plt.show()

if __name__ == "__main__":
    femaleDat = "../results/six_methods/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/six_methods/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/six_methods/clusters/male_clusters.pdf"
    femalePDF = "../plots/six_methods/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/six_methods/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/six_methods/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/fasttext/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/fasttext/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/fasttext/clusters/male_clusters.pdf"
    femalePDF = "../plots/fasttext/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/fasttext/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/fasttext/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/openAI/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/openAI/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/openAI/clusters/male_clusters.pdf"
    femalePDF = "../plots/openAI/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/openAI/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/openAI/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/cohere/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/cohere/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/cohere/clusters/male_clusters.pdf"
    femalePDF = "../plots/cohere/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/cohere/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/cohere/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/google/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/google/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/google/clusters/male_clusters.pdf"
    femalePDF = "../plots/google/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/google/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/google/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/microsoft/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/microsoft/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/microsoft/clusters/male_clusters.pdf"
    femalePDF = "../plots/microsoft/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/microsoft/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/microsoft/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/BGE/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/BGE/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/BGE/clusters/male_clusters.pdf"
    femalePDF = "../plots/BGE/clusters/female_clusters.pdf"
    maleClusterToTopic = "../results/BGE/clusters/male_clusters_to_topics.json"
    femaleClusterToTopic = "../results/BGE/clusters/female_clusters_to_topics.json"

    plot(maleDat, malePDF, maleClusterToTopic, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, femaleClusterToTopic, "Female Scatter Plot of Words by Cluster")
