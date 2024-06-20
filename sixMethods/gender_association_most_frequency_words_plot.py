import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(input, output, num_columns_list):
    print(f"\n == {input} == \n")
    data = pd.read_csv(input)

    female_percentages = []
    male_percentages = []

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


    bar_width = 0.35
    N = len(num_columns_list)
    x = np.arange(N)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot
    ax.bar(x - bar_width / 2, female_percentages, bar_width, label='Female')
    ax.bar(x + bar_width / 2, male_percentages, bar_width, label='Male')

    # Add labels and title
    ax.set_xlabel('Gender Association')
    ax.set_ylabel('N Most Frequent Words')
    ax.set_title('Gender Association by Frequency Range')
    ax.set_xticks(x)
    ax.set_xticklabels(num_columns_list)
    ax.legend()

    # Display the plot
    # Save the plot as a PDF file
    plt.savefig(output, format='pdf')


if __name__ == "__main__":
    num_columns_list = [100, 1000, 10000, 100000]
    # plot("../results/six_methods/most_frequency_words/glove_100000_most_frequency.csv", "../plots/six_methods/most_frequency_words/glove_frequency_ratio.pdf", num_columns_list)
    # plot("../results/openAI/most_frequency_words/openAI_100000_most_frequency.csv", "../plots/openAI/most_frequency_words/openai_frequency_ratio.pdf", num_columns_list)
    # plot("../results/fasttext/most_frequency_words/ft_100000_most_frequency.csv",
    #      "../plots/fasttext/most_frequency_words/ft_frequency_ratio.pdf", num_columns_list)
    # plot("../results/cohere/most_frequency_words/cohere_100000_most_frequency.csv",
    #      "../plots/cohere/most_frequency_words/cohere_frequency_ratio.pdf", num_columns_list)
    plot("../results/google/most_frequency_words/google_100000_most_frequency.csv",
         "../plots/google/most_frequency_words/google_frequency_ratio.pdf", num_columns_list)
