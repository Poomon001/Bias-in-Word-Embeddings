'''
plot_cluster
Visualize word clusters derived from word embeddings. The word embeddings are reduced to two dimensions using the t-SNE
(t-Distributed Stochastic Neighbor Embedding) method to facilitate visualization.
'''

import pandas as pd
import matplotlib.pyplot as plt

def writeTo(filename, data, words):
    wordCount = 0
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("word,female_effect_size,female_p_value\n")
        for index, row in data.iterrows():
            if wordCount < words:
                file.write(f"{row['word']},{row['female_effect_size']},{row['female_p_value']}\n")
            else:
                break
            wordCount += 1

def plot(dat, pdf, title):
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

    # Add a color bar
    plt.colorbar(scatter, label='Cluster')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the plot as a PDF
    plt.savefig(pdf, format='pdf')

if __name__ == "__main__":
    femaleDat = "../results/six_methods/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/six_methods/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/six_methods/clusters/male_clusters.pdf"
    femalePDF = "../plots/six_methods/clusters/female_clusters.pdf"

    # plot(maleDat, malePDF, "Male Scatter Plot of Words by Cluster")
    # plot(femaleDat, femalePDF,"Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/fasttext/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/fasttext/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/fasttext/clusters/male_clusters.pdf"
    femalePDF = "../plots/fasttext/clusters/female_clusters.pdf"

    # plot(maleDat, malePDF, "Male Scatter Plot of Words by Cluster")
    # plot(femaleDat, femalePDF, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/openAI/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/openAI/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/openAI/clusters/male_clusters.pdf"
    femalePDF = "../plots/openAI/clusters/female_clusters.pdf"

    # plot(maleDat, malePDF, "Male Scatter Plot of Words by Cluster")
    # plot(femaleDat, femalePDF, "Female Scatter Plot of Words by Cluster")

    # femaleDat = "../results/cohere/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    # maleDat = "../results/cohere/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    # malePDF = "../plots/cohere/clusters/male_clusters.pdf"
    # femalePDF = "../plots/cohere/clusters/female_clusters.pdf"

    # plot(maleDat, malePDF, "Male Scatter Plot of Words by Cluster")
    # plot(femaleDat, femalePDF, "Female Scatter Plot of Words by Cluster")

    femaleDat = "../results/google/clusters/tsne_clusters_female_male_vis_elkan_11.dat"
    maleDat = "../results/google/clusters/tsne_clusters_male_male_vis_elkan_11.dat"
    malePDF = "../plots/google/clusters/male_clusters.pdf"
    femalePDF = "../plots/google/clusters/female_clusters.pdf"

    plot(maleDat, malePDF, "Male Scatter Plot of Words by Cluster")
    plot(femaleDat, femalePDF, "Female Scatter Plot of Words by Cluster")