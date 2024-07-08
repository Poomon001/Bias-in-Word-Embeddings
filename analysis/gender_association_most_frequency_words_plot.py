import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(input, output, num_columns_list, name):
    print(f"\n == {input} == \n")
    data = pd.read_csv(input)

    female_percentages = []
    male_percentages = []
    female_counts = []
    male_counts = []

    for num_columns in num_columns_list:
        n_data = data[:num_columns]
        # Filter rows where female_effect_size is positive and male_effect_size is negative
        female_data = n_data[n_data['female_effect_size'] > 0]
        male_data = n_data[n_data['female_effect_size'] < 0]

        # Count the number of occurrences for each word
        total_count = len(female_data) + len(male_data)
        female_count = len(female_data)
        male_count = len(male_data)

        print("Number of N:", num_columns)
        print("Total:", total_count)
        print("Female:", female_count)
        print("Male:", male_count)
        print("\n")

        female_percentage = (female_count / total_count) * 100
        male_percentage = (male_count / total_count) * 100

        female_percentages.append(female_percentage)
        male_percentages.append(male_percentage)
        female_counts.append(female_count)
        male_counts.append(male_count)


    # Plotting the data
    plt.figure(figsize=(10, 6))

    bar_width = 0.35
    indices = range(len(num_columns_list))

    plt.bar(indices, female_percentages, bar_width, label='Female %', color='pink')
    plt.bar([i + bar_width for i in indices], male_percentages, bar_width, label='Male %', color='lightblue')

    plt.xlabel('Number of Columns')
    plt.ylabel('Percentage')
    plt.title(f'Female and Male Effect Sizes From {name}')
    plt.xticks([i + bar_width / 2 for i in indices], num_columns_list)
    plt.legend()

    # counts on the bars
    for i, (female_count, male_count) in enumerate(zip(female_counts, male_counts)):
        plt.text(i, female_percentages[i] + 1, str(female_count), ha='center', va='bottom')
        plt.text(i + bar_width, male_percentages[i] + 1, str(male_count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.show()


if __name__ == "__main__":
    num_columns_list = [100, 1000, 10000, 100000]
    input = "../results/glove/most_frequency_words/glove_gender_100000_most_frequency.csv"
    output = "../plots/glove/most_frequency_words/glove_frequency_ratio.pdf"
    name = "GloVe"
    # plot(input, output, num_columns_list, name)

    input = "../results/openAI/most_frequency_words/openAI_gender_100000_most_frequency.csv"
    output =  "../plots/openAI/most_frequency_words/openai_frequency_ratio.pdf"
    name = "OpenAI"
    # plot(input, output, num_columns_list, name)

    input = "../results/fasttext/most_frequency_words/ft_gender_100000_most_frequency.csv"
    output = "../plots/fasttext/most_frequency_words/ft_frequency_ratio.pdf"
    name = "FastText"
    # plot(input, output, num_columns_list, name)

    input = "../results/cohere/most_frequency_words/cohere_100000_most_frequency.csv"
    output = "../plots/cohere/most_frequency_words/cohere_frequency_ratio.pdf"
    name = "Cohere"
    # plot(input, output, num_columns_list, name)

    input = "../results/google/most_frequency_words/google_100000_most_frequency.csv"
    output = "../plots/google/most_frequency_words/google_frequency_ratio.pdf"
    name = "Google"
    # plot(input, output, num_columns_list, name)

    input = "../results/microsoft/most_frequency_words/microsoft_100000_most_frequency.csv"
    output = "../plots/microsoft/most_frequency_words/microsoft_frequency_ratio.pdf"
    name = "Microsoft"
    # plot(input, output, num_columns_list, name)

    input = "../results/BGE/most_frequency_words/BGE_100000_most_frequency.csv"
    output = "../plots/BGE/most_frequency_words/BGE_frequency_ratio.pdf"
    name = "BGE"
    # plot(input, output, num_columns_list, name)
