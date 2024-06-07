import pandas as pd
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

def process(top_1k_file, tsneCSV, tsneDAT, pdf):
    top_1k = pd.read_csv(top_1k_file, na_values=None, keep_default_na=False,names=['word','female_effect_size','p_value'],skiprows=1)
    target_words = top_1k['word'].tolist()

    embedding_df = pd.read_csv("../raw/glove_1000_most_freq_skip.txt",sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=1000)

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
    plt.colorbar()
    plt.savefig(pdf, format='pdf')

if __name__ == "__main__":
    tsneCSV = "../results/six_methods/clusters/tsne_dims_1k.csv"
    tsneDAT = "../results/six_methods/clusters/tsne_vis_1k.dat"
    pdf = "../plots/six_methods/tsne_top_1k/tsne.pdf"
    process("../results/six_methods/most_frequency_words/glove_1000_most_frequency.csv", tsneCSV, tsneDAT, pdf)