import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot(ratio, model, output):
    ratio = ratio.drop(columns=['num_words'])
    ratio.plot(kind='bar')
    plt.title(f'Frequency Ratio: {model}')
    plt.xlabel('Gender_effect size')
    plt.ylabel('Number of Words')

    plt.savefig(output)
    plt.show()


def process(embeddingInoutPath, resultOutputPath):
    source_df = pd.read_csv(embeddingInoutPath, na_values=None, keep_default_na=False)

    # Female vs. Male ratio by frequency range, effect size range
    # frequency_ceilings = [100, 1000, 10000, 100000]
    frequency_ceilings = [88]
    effect_size_floors = [0, .2, .5, .8]

    es_list = []

    for ceiling in frequency_ceilings:
        head_df = source_df.head(ceiling)
        ceiling_counts = [ceiling]

        print(head_df)
        for es in effect_size_floors:
            es_df = head_df.loc[head_df['female_effect_size'] >= es]
            es_quantity = len(es_df.index.tolist())
            ceiling_counts.append(es_quantity)

        for es in effect_size_floors:
            es_df = head_df.loc[head_df['female_effect_size'] <= -es]
            es_quantity = len(es_df.index.tolist())
            ceiling_counts.append(es_quantity)

        es_list.append(ceiling_counts)

    es_arr = np.array(es_list)
    cols = ['num_words'] + [f'female_{str(i)}' for i in effect_size_floors] + [f'male_{str(i)}' for i in effect_size_floors]
    es_df = pd.DataFrame(es_arr, columns=cols)
    es_df.to_csv(resultOutputPath)
    return es_df

if __name__ == "__main__":
    es_df = process("results/sc-weat/openAI_vad_words.csv", "results/frequency/openAI_female_male_ratios.csv")
    plot(es_df, "OpenAI", "plots/frequency/OpenAI_frequency_ratio.png")

    es_df = process("results/sc-weat/glove_vad_words.csv", "results/frequency/glove_female_male_ratios.csv")
    plot(es_df, "glove", "plots/frequency/glove_frequency_ratio.png")

    es_df = process("results/sc-weat/ft_vad_words.csv", "results/frequency/ft_female_male_ratios.csv")
    plot(es_df, "ft", "plots/frequency/ft_frequency_ratio.png")
