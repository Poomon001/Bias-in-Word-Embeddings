import pandas as pd
import json

def replace_clusters(text, cluster_map):
    for cluster_num, title in cluster_map.items():
        text = text.replace(f'Cluster {cluster_num}:', f'Cluster {title}:')
    return text

def process(clusters_to_topics_file, input, output):
    # Read text file
    with open(input, 'r') as file:
        text_content = file.read()

    # Load JSON mapping cluster numbers to titles
    with open(clusters_to_topics_file, 'r') as json_file:
        cluster_map = json.load(json_file)

    # Replace clusters in the text content
    modified_text = replace_clusters(text_content, cluster_map)

    # Write back to the text file
    with open(output, 'w') as file:
        file.write(modified_text)

if __name__ == "__main__":
    clusterToTopic = "../results/cohere/clusters/male_clusters_to_topics.json"
    input = "../results/cohere/clusters/male_clusters_11.txt"
    output = "../results/cohere/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/cohere/clusters/female_clusters_to_topics.json"
    input = "../results/cohere/clusters/female_clusters_11.txt"
    output = "../results/cohere/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/fasttext/clusters/male_clusters_to_topics.json"
    input = "../results/fasttext/clusters/male_clusters_11.txt"
    output = "../results/fasttext/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/fasttext/clusters/female_clusters_to_topics.json"
    input = "../results/fasttext/clusters/female_clusters_11.txt"
    output = "../results/fasttext/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)


    clusterToTopic = "../results/BGE/clusters/male_clusters_to_topics.json"
    input = "../results/BGE/clusters/male_clusters_11.txt"
    output = "../results/BGE/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/BGE/clusters/female_clusters_to_topics.json"
    input = "../results/BGE/clusters/female_clusters_11.txt"
    output = "../results/BGE/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/fasttext/clusters/male_clusters_to_topics.json"
    input = "../results/fasttext/clusters/male_clusters_11.txt"
    output = "../results/fasttext/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/fasttext/clusters/female_clusters_to_topics.json"
    input = "../results/fasttext/clusters/female_clusters_11.txt"
    output = "../results/fasttext/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/google/clusters/male_clusters_to_topics.json"
    input = "../results/google/clusters/male_clusters_11.txt"
    output = "../results/google/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/google/clusters/female_clusters_to_topics.json"
    input = "../results/google/clusters/female_clusters_11.txt"
    output = "../results/google/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/microsoft/clusters/male_clusters_to_topics.json"
    input = "../results/microsoft/clusters/male_clusters_11.txt"
    output = "../results/microsoft/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/microsoft/clusters/female_clusters_to_topics.json"
    input = "../results/microsoft/clusters/female_clusters_11.txt"
    output = "../results/microsoft/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/six_methods/clusters/male_clusters_to_topics.json"
    input = "../results/six_methods/clusters/male_clusters_11.txt"
    output = "../results/six_methods/clusters/male_clusters_to_topics.txt"
    process(clusterToTopic, input, output)

    clusterToTopic = "../results/six_methods/clusters/female_clusters_to_topics.json"
    input = "../results/six_methods/clusters/female_clusters_11.txt"
    output = "../results/six_methods/clusters/female_clusters_to_topics.txt"
    process(clusterToTopic, input, output)