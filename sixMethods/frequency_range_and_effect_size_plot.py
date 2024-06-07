import pandas as pd
from tabulate import tabulate

def plot(input, output, num_columns_list):
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
    print(es_list)

    # Headers
    headers = ["N-words", "d > 0.0 Female", "d > 0.2 Female", "d > 0.5 Female", "d > 0.8 Female", "d > 0.0 Male", "d > 0.2 Male", "d > 0.5 Male", "d > 0.8 Male"]
    table = tabulate(es_list, headers, tablefmt="grid")
    print(table)



if __name__ == "__main__":
    num_columns_list = [100, 1000, 10000, 100000]
    # plot("../results/six_methods/most_frequency_words/glove_100000_most_frequency.csv",
    #      "../plots/six_methods/frequency_range_and_effect_size/glove_effect_size_table.png", num_columns_list)
    plot("../results/openAI/openAI_100000_most_frequency.csv",
         "../plots/openAI/frequency_range_and_effect_size/glove_effect_size_table.png", num_columns_list)