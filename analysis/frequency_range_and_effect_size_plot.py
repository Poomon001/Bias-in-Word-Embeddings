import pandas as pd
import numpy as np

def plot(input, output, num_columns_list, groups):
    print(f"\n == {input} == \n")
    data = pd.read_csv(input, na_values=None, keep_default_na=False)

    effect_size_floors = [0, .2, .5, .8]

    # for 100, 1000, 10000 N-words
    es_list = []

    for ceiling in num_columns_list:
        head_df = data.head(ceiling)
        ceiling_counts = [ceiling]

        for es in effect_size_floors:
            es_df = head_df.loc[head_df[f'{groups[0]}_effect_size'] >= es]
            es_quantity = len(es_df.index.tolist())
            ceiling_counts.append(es_quantity)

        for es in effect_size_floors:
            es_df = head_df.loc[head_df[f'{groups[0]}_effect_size'] <= -es]
            es_quantity = len(es_df.index.tolist())
            ceiling_counts.append(es_quantity)

        es_list.append(ceiling_counts)

    es_arr = np.array(es_list)
    cols = ['num_words'] + [f'{groups[0]}_{str(i)}' for i in effect_size_floors] + [f'{groups[1]}_{str(i)}' for i in effect_size_floors]
    es_df = pd.DataFrame(es_arr, columns=cols)
    es_df.to_csv(output)



if __name__ == "__main__":
    num_columns_list = [100, 1000, 10000, 100000]
    groups = ["female", "male"]

    plot("../results/glove/most_frequency_words/glove_gender_100000_most_frequency.csv",
         "../results/glove/frequency_analysis/glove_gender_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/fasttext/most_frequency_words/ft_gender_100000_most_frequency.csv",
         "../results/fasttext/frequency_analysis/ft_gender_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/openAI/most_frequency_words/openAI_gender_100000_most_frequency.csv",
         "../results/openAI/frequency_analysis/openai_gender_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/cohere/most_frequency_words/cohere_gender_100000_most_frequency.csv",
         "../results/cohere/frequency_analysis/cohere_gender_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/google/most_frequency_words/google_gender_100000_most_frequency.csv",
        "../results/google/frequency_analysis/google_gender_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/microsoft/most_frequency_words/microsoft_gender_100000_most_frequency.csv",
         "../results/microsoft/frequency_analysis/microsoft_gender_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/BGE/most_frequency_words/BGE_gender_100000_most_frequency.csv",
         "../results/BGE/frequency_analysis/BGE_gender_frequency_analysis.csv",
         num_columns_list, groups)

    print("Finish gender class process")

    groups = ["caucasian", "black"]

    plot("../results/glove/most_frequency_words/glove_race_WB_100000_most_frequency.csv",
         "../results/glove/frequency_analysis/glove_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/fasttext/most_frequency_words/ft_race_WB_100000_most_frequency.csv",
         "../results/fasttext/frequency_analysis/ft_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/openAI/most_frequency_words/openAI_race_WB_100000_most_frequency.csv",
         "../results/openAI/frequency_analysis/openai_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/cohere/most_frequency_words/cohere_race_WB_100000_most_frequency.csv",
         "../results/cohere/frequency_analysis/cohere_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/google/most_frequency_words/google_race_WB_100000_most_frequency.csv",
         "../results/google/frequency_analysis/google_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/microsoft/most_frequency_words/microsoft_race_WB_100000_most_frequency.csv",
         "../results/microsoft/frequency_analysis/microsoft_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/BGE/most_frequency_words/BGE_race_WB_100000_most_frequency.csv",
         "../results/BGE/frequency_analysis/BGE_race_WB_frequency_analysis.csv",
         num_columns_list, groups)

    print("Finish first race class process")

    groups = ["caucasian", "asian"]

    plot("../results/glove/most_frequency_words/glove_race_WA_100000_most_frequency.csv",
         "../results/glove/frequency_analysis/glove_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/fasttext/most_frequency_words/ft_race_WA_100000_most_frequency.csv",
         "../results/fasttext/frequency_analysis/ft_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/openAI/most_frequency_words/openAI_race_WA_100000_most_frequency.csv",
         "../results/openAI/frequency_analysis/openai_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/cohere/most_frequency_words/cohere_race_WA_100000_most_frequency.csv",
         "../results/cohere/frequency_analysis/cohere_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/google/most_frequency_words/google_race_WA_100000_most_frequency.csv",
         "../results/google/frequency_analysis/google_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/microsoft/most_frequency_words/microsoft_race_WA_100000_most_frequency.csv",
         "../results/microsoft/frequency_analysis/microsoft_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/BGE/most_frequency_words/BGE_race_WA_100000_most_frequency.csv",
         "../results/BGE/frequency_analysis/BGE_race_WA_frequency_analysis.csv",
         num_columns_list, groups)

    print("Finish second race class process")

    groups = ["asian", "black"]

    plot("../results/glove/most_frequency_words/glove_race_AB_100000_most_frequency.csv",
         "../results/glove/frequency_analysis/glove_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/fasttext/most_frequency_words/ft_race_AB_100000_most_frequency.csv",
         "../results/fasttext/frequency_analysis/ft_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/openAI/most_frequency_words/openAI_race_AB_100000_most_frequency.csv",
         "../results/openAI/frequency_analysis/openai_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/cohere/most_frequency_words/cohere_race_AB_100000_most_frequency.csv",
         "../results/cohere/frequency_analysis/cohere_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/google/most_frequency_words/google_race_AB_100000_most_frequency.csv",
         "../results/google/frequency_analysis/google_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/microsoft/most_frequency_words/microsoft_race_AB_100000_most_frequency.csv",
         "../results/microsoft/frequency_analysis/microsoft_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    plot("../results/BGE/most_frequency_words/BGE_race_AB_100000_most_frequency.csv",
         "../results/BGE/frequency_analysis/BGE_race_AB_frequency_analysis.csv",
         num_columns_list, groups)

    print("Finish third race class process")