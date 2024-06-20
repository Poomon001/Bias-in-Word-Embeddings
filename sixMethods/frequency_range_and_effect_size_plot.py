import pandas as pd
from tabulate import tabulate
import numpy as np

def plot(input, output, img, num_columns_list):
    print(f"\n == {input} == \n")
    data = pd.read_csv(input)

    effect_size_floors = [0, .2, .5, .8]

    # for 100, 1000, 10000 N-words
    #[[0.0 female, 0.2 female, 0.5 female, 0.8 female, 0.0 male, 0.2 male, 0.5 male, 0.8 male],
    # [0.0 female, 0.2 female, 0.5 female, 0.8 female, 0.0 male, 0.2 male, 0.5 male, 0.8 male],
    # [0.0 female, 0.2 female, 0.5 female, 0.8 female, 0.0 male, 0.2 male, 0.5 male, 0.8 male]]
    es_list = []
    for num_columns in num_columns_list:
        frequency_ceilings = [num_columns]

        for ceiling in frequency_ceilings:
            n_data = data[:num_columns]
            ceiling_counts = [ceiling]

            for es in effect_size_floors:
                es_df = n_data.loc[n_data['female_effect_size'] >= es]
                es_quantity = len(es_df.index.tolist())
                ceiling_counts.append(es_quantity)

            for es in effect_size_floors:
                es_df = n_data.loc[n_data['female_effect_size'] <= -es]
                es_quantity = len(es_df.index.tolist())
                ceiling_counts.append(es_quantity)

        es_list.append(ceiling_counts)

    es_arr = np.array(es_list)
    cols = ['num_words'] + [f'female_{str(i)}' for i in effect_size_floors] + [f'male_{str(i)}' for i in effect_size_floors]
    es_df = pd.DataFrame(es_arr, columns=cols)
    es_df.to_csv(output)



if __name__ == "__main__":
    num_columns_list = [100, 1000, 10000, 100000]

    # input = "../results/six_methods/most_frequency_words/glove_100000_most_frequency.csv"
    # output = "../results/six_methods/frequency_analysis/glove_frequency_analysis.csv"
    # img = "../plots/six_methods/frequency_range_and_effect_size/glove_effect_size_table.png"
    # plot(input, output, img, num_columns_list)
    #
    # input = "../results/openAI/most_frequency_words/openAI_100000_most_frequency.csv"
    # output = "../results/openAI/frequency_analysis/openai_frequency_analysis.csv"
    # img = "../plots/openAI/frequency_range_and_effect_size/glove_effect_size_table.png"
    # plot(input, output, img, num_columns_list)
    #
    # input = "../results/fasttext/most_frequency_words/ft_100000_most_frequency.csv"
    # output = "../results/fasttext/frequency_analysis/ft_frequency_analysis.csv"
    # img = "../plots/fasttext/frequency_range_and_effect_size/ft_effect_size_table.png"
    # plot(input, output, img, num_columns_list)

    # input = "../results/cohere/most_frequency_words/cohere_100000_most_frequency.csv"
    # output = "../results/cohere/frequency_analysis/cohere_frequency_analysis.csv"
    # img = "../plots/cohere/frequency_range_and_effect_size/cohere_effect_size_table.png"
    # plot(input, output, img, num_columns_list)

    input = "../results/google/most_frequency_words/google_100000_most_frequency.csv"
    output = "../results/google/frequency_analysis/google_frequency_analysis.csv"
    img = "../plots/google/frequency_range_and_effect_size/google_effect_size_table.png"
    plot(input, output, img, num_columns_list)