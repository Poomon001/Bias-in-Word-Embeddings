import pandas as pd
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

def process(embedding_100k_file, top_1k_file, tsneCSV, tsneDAT, pdf):
    top_1k = pd.read_csv(top_1k_file, na_values=None, keep_default_na=False,names=['word','female_effect_size','p_value'],skiprows=1)
    target_words = top_1k['word'].tolist()

    # 100k word embedded file
    embedding_df = pd.read_csv(embedding_100k_file,sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=1000)

    # create a data frame from target word based on word embedding
    target_df = embedding_df.loc[target_words]
    target_data = target_df.to_numpy()
    print(target_data.shape)

    reduced_dims = TSNE().fit_transform(target_data)
    tsne_df = pd.DataFrame(reduced_dims, index=target_words, columns=['x', 'y'])
    tsne_df.to_csv(tsneCSV)
    print(tsne_df)

    tsne_x = tsne_df['x'].tolist()
    tsne_y = tsne_df['y'].tolist()
    top_es = top_1k['female_effect_size'].tolist()
    print(top_es)

    write_string = 'x\ty\teffect_size\n' + '\n'.join(['\t'.join([str(tsne_x[i]), str(tsne_y[i]), str(top_es[i])]) for i in range(len(tsne_x))])
    with open(tsneDAT, 'w') as writer:
        writer.write(write_string)

    plt.scatter(tsne_df['x'].tolist(), tsne_df['y'].tolist(), c=top_1k['female_effect_size'].tolist())

    # Save the plot as a PDF file
    plt.colorbar(label='Female Effect Size')
    plt.savefig(pdf, format='pdf')

    plt.show()

if __name__ == "__main__":
    embedding_100k_file = "../raw/glove_100000_most_freq_skip.txt"
    top_1k = "../results/six_methods/most_frequency_words/glove_1000_most_frequency.csv"
    tsneCSV = "../results/six_methods/clusters/tsne_dims_1k.csv"
    tsneDAT = "../results/six_methods/clusters/tsne_vis_1k.dat"
    pdf = "../plots/six_methods/tsne_top_1k/tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf)

    embedding_100k_file = "../openAI/openAI_100000_most_freq_skip.txt"
    top_1k = "../results/openAI/most_frequency_words/openAI_1000_most_frequency.csv"
    tsneCSV = "../results/openAI/clusters/tsne_dims_1k.csv"
    tsneDAT = "../results/openAI/clusters/tsne_vis_1k.dat"
    pdf = "../plots/openAI/tsne_top_1k/tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf)

    embedding_100k_file = "../raw/ft_100000_most_freq_skip.csv"
    top_1k = "../results/fasttext/most_frequency_words/ft_1000_most_frequency.csv"
    tsneCSV = "../results/fasttext/clusters/tsne_dims_1k.csv"
    tsneDAT = "../results/fasttext/clusters/tsne_vis_1k.dat"
    pdf = "../plots/fasttext/tsne_top_1k/tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf)

    embedding_100k_file = "../cohere/cohere_100000_most_freq_skip.txt"
    top_1k = "../results/cohere/most_frequency_words/cohere_1000_most_frequency.csv"
    tsneCSV = "../results/cohere/clusters/tsne_dims_1k.csv"
    tsneDAT = "../results/cohere/clusters/tsne_vis_1k.dat"
    pdf = "../plots/cohere/tsne_top_1k/tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf)

    embedding_100k_file = "../google/google_100000_most_freq_skip.txt"
    top_1k = "../results/google/most_frequency_words/google_1000_most_frequency.csv"
    tsneCSV = "../results/google/clusters/tsne_dims_1k.csv"
    tsneDAT = "../results/google/clusters/tsne_vis_1k.dat"
    pdf = "../plots/google/tsne_top_1k/tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf)
