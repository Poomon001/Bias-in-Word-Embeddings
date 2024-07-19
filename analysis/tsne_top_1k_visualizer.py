import pandas as pd
import csv
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

def process(embedding_100k_file, top_1k_file, tsneCSV, tsneDAT, pdf, groups):
    top_1k = pd.read_csv(top_1k_file, na_values=None, keep_default_na=False,names=['word',f'{groups[0]}_effect_size','p_value'],skiprows=1)
    target_words = top_1k['word'].tolist()

    # 100k word embedded file
    embedding_df = pd.read_csv(embedding_100k_file,sep=' ',header=None,index_col=0, na_values=None, keep_default_na=False,quoting=csv.QUOTE_NONE,nrows=1000)

    # create a data frame from target word based on word embedding
    target_df = embedding_df.loc[target_words] # TODO: if KeyError: "['TRUE'] not in index", then try to change the first 'TRUE' word -> 'true' word in .csv file
    target_data = target_df.to_numpy()
    print(target_data.shape)

    reduced_dims = TSNE().fit_transform(target_data)
    tsne_df = pd.DataFrame(reduced_dims, index=target_words, columns=['x', 'y'])
    tsne_df.to_csv(tsneCSV)
    print(tsne_df)

    tsne_x = tsne_df['x'].tolist()
    tsne_y = tsne_df['y'].tolist()
    top_es = top_1k[f'{groups[0]}_effect_size'].tolist()
    print(top_es)

    write_string = 'x\ty\teffect_size\n' + '\n'.join(['\t'.join([str(tsne_x[i]), str(tsne_y[i]), str(top_es[i])]) for i in range(len(tsne_x))])
    with open(tsneDAT, 'w') as writer:
        writer.write(write_string)

    plt.scatter(tsne_df['x'].tolist(), tsne_df['y'].tolist(), c=top_1k[f'{groups[0]}_effect_size'].tolist())

    # Save the plot as a PDF file
    plt.colorbar(label=f'{groups[0].title()} Effect Size')
    plt.savefig(pdf, format='pdf')

    plt.show()

if __name__ == "__main__":
    groups = ["female", "male"]
    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_1k = "../results/glove/most_frequency_words/glove_gender_1000_most_frequency.csv"
    tsneCSV = "../results/glove/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/glove/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/glove/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_1k = "../results/fasttext/most_frequency_words/ft_gender_1000_most_frequency.csv"
    tsneCSV = "../results/fasttext/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/fasttext/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/fasttext/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_1k = "../results/openAI/most_frequency_words/openAI_gender_1000_most_frequency.csv"
    tsneCSV = "../results/openAI/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/openAI/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/openAI/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_1k = "../results/cohere/most_frequency_words/cohere_gender_1000_most_frequency.csv"
    tsneCSV = "../results/cohere/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/cohere/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/cohere/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_1k = "../results/google/most_frequency_words/google_gender_1000_most_frequency.csv"
    tsneCSV = "../results/google/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/google/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/google/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_1k = "../results/microsoft/most_frequency_words/microsoft_gender_1000_most_frequency.csv"
    tsneCSV = "../results/microsoft/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/microsoft/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/microsoft/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_1k = "../results/BGE/most_frequency_words/BGE_gender_1000_most_frequency.csv"
    tsneCSV = "../results/BGE/clusters/gender_tsne_dims_1k.csv"
    tsneDAT = "../results/BGE/clusters/gender_tsne_vis_1k.dat"
    pdf = "../plots/BGE/tsne_top_1k/gender_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    print("Finish gender class process")

    groups = ["caucasian", "black"]
    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_1k = "../results/glove/most_frequency_words/glove_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/glove/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/glove/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/glove/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_1k = "../results/fasttext/most_frequency_words/ft_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/fasttext/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/fasttext/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/fasttext/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_1k = "../results/openAI/most_frequency_words/openAI_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/openAI/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/openAI/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/openAI/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_1k = "../results/cohere/most_frequency_words/cohere_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/cohere/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/cohere/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/cohere/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_1k = "../results/google/most_frequency_words/google_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/google/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/google/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/google/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_1k = "../results/microsoft/most_frequency_words/microsoft_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/microsoft/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/microsoft/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/microsoft/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_1k = "../results/BGE/most_frequency_words/BGE_race_WB_1000_most_frequency.csv"
    tsneCSV = "../results/BGE/clusters/race_WB_tsne_dims_1k.csv"
    tsneDAT = "../results/BGE/clusters/race_WB_tsne_vis_1k.dat"
    pdf = "../plots/BGE/tsne_top_1k/race_WB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    print("Finish first race class process")

    groups = ["caucasian", "asian"]

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_1k = "../results/glove/most_frequency_words/glove_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/glove/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/glove/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/glove/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_1k = "../results/fasttext/most_frequency_words/ft_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/fasttext/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/fasttext/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/fasttext/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_1k = "../results/openAI/most_frequency_words/openAI_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/openAI/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/openAI/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/openAI/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_1k = "../results/cohere/most_frequency_words/cohere_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/cohere/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/cohere/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/cohere/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_1k = "../results/google/most_frequency_words/google_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/google/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/google/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/google/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_1k = "../results/microsoft/most_frequency_words/microsoft_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/microsoft/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/microsoft/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/microsoft/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_1k = "../results/BGE/most_frequency_words/BGE_race_WA_1000_most_frequency.csv"
    tsneCSV = "../results/BGE/clusters/race_WA_tsne_dims_1k.csv"
    tsneDAT = "../results/BGE/clusters/race_WA_tsne_vis_1k.dat"
    pdf = "../plots/BGE/tsne_top_1k/race_WA_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    print("Finish second race class process")

    groups = ["female", "black"]
    embedding_100k_file = "D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    top_1k = "../results/glove/most_frequency_words/glove_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/glove/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/glove/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/glove/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    top_1k = "../results/fasttext/most_frequency_words/ft_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/fasttext/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/fasttext/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/fasttext/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt"
    top_1k = "../results/openAI/most_frequency_words/openAI_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/openAI/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/openAI/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/openAI/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    top_1k = "../results/cohere/most_frequency_words/cohere_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/cohere/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/cohere/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/cohere/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    top_1k = "../results/google/most_frequency_words/google_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/google/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/google/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/google/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt"
    top_1k = "../results/microsoft/most_frequency_words/microsoft_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/microsoft/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/microsoft/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/microsoft/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    embedding_100k_file = "D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt"
    top_1k = "../results/BGE/most_frequency_words/BGE_race_AB_1000_most_frequency.csv"
    tsneCSV = "../results/BGE/clusters/race_AB_tsne_dims_1k.csv"
    tsneDAT = "../results/BGE/clusters/race_AB_tsne_vis_1k.dat"
    pdf = "../plots/BGE/tsne_top_1k/race_AB_tsne.pdf"
    process(embedding_100k_file, top_1k, tsneCSV, tsneDAT, pdf, groups)

    print("Finish third race class process")

