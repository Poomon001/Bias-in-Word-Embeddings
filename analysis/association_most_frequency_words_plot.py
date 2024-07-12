import pandas as pd
import matplotlib.pyplot as plt

def plot(input, output, num_columns_list, name, groups):
    print(f"\n == {input} == \n")
    data = pd.read_csv(input)

    first_group_percentages = []
    second_group_percentages = []
    first_group_counts = []
    second_group_counts = []

    for num_columns in num_columns_list:
        n_data = data[:num_columns]
        # Filter rows where {first_group}_effect_size is positive and {second_group}_effect_size is negative
        first_group_data = n_data[n_data[f'{groups[0]}_effect_size'] > 0]
        second_group_data = n_data[n_data[f'{groups[0]}_effect_size'] < 0]

        # Count the number of occurrences for each word
        total_count = len(first_group_data) + len(second_group_data)
        first_group_count = len(first_group_data)
        second_group_count = len(second_group_data)

        print("Number of N:", num_columns)
        print("Total:", total_count)
        print(f"{groups[0].title()}:", first_group_count)
        print(f"{groups[1].title()}:", second_group_count)
        print("\n")

        first_group_percentage = (first_group_count / total_count) * 100
        second_group_percentage = (second_group_count / total_count) * 100

        first_group_percentages.append(first_group_percentage)
        second_group_percentages.append(second_group_percentage)
        first_group_counts.append(first_group_count)
        second_group_counts.append(second_group_count)


    # Plotting the data
    plt.figure(figsize=(10, 6))

    bar_width = 0.35
    indices = range(len(num_columns_list))

    plt.bar(indices, first_group_percentages, bar_width, label=f'{groups[0]} %', color='pink')
    plt.bar([i + bar_width for i in indices], second_group_percentages, bar_width, label=f'{groups[1]} %', color='lightblue')

    plt.xlabel('Number of Columns')
    plt.ylabel('Percentage')
    plt.title(f'{groups[0].title()} and {groups[1].title()} Effect Sizes From {name}')
    plt.xticks([i + bar_width / 2 for i in indices], num_columns_list)
    plt.legend()

    # counts on the bars
    for i, (first_group_count, second_group_count) in enumerate(zip(first_group_counts, second_group_counts)):
        plt.text(i, first_group_percentages[i] + 1, str(first_group_count), ha='center', va='bottom')
        plt.text(i + bar_width, second_group_percentages[i] + 1, str(second_group_count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.show()


if __name__ == "__main__":
    num_columns_list = [100, 1000, 10000, 100000]
    groups = ["female", "male"]

    input = "../results/glove/most_frequency_words/glove_gender_100000_most_frequency.csv"
    output = "../plots/glove/most_frequency_words/glove_gender_frequency_ratio.pdf"
    name = "GloVe"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/openAI/most_frequency_words/openAI_gender_100000_most_frequency.csv"
    output =  "../plots/openAI/most_frequency_words/openai_gender_frequency_ratio.pdf"
    name = "OpenAI"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/fasttext/most_frequency_words/ft_gender_100000_most_frequency.csv"
    output = "../plots/fasttext/most_frequency_words/ft_gender_frequency_ratio.pdf"
    name = "FastText"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/cohere/most_frequency_words/cohere_gender_100000_most_frequency.csv"
    output = "../plots/cohere/most_frequency_words/cohere_gender_frequency_ratio.pdf"
    name = "Cohere"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/google/most_frequency_words/google_gender_100000_most_frequency.csv"
    output = "../plots/google/most_frequency_words/google_gender_frequency_ratio.pdf"
    name = "Google"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/microsoft/most_frequency_words/microsoft_gender_100000_most_frequency.csv"
    output = "../plots/microsoft/most_frequency_words/microsoft_gender_frequency_ratio.pdf"
    name = "Microsoft"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/BGE/most_frequency_words/BGE_gender_100000_most_frequency.csv"
    output = "../plots/BGE/most_frequency_words/BGE_gender_frequency_ratio.pdf"
    name = "BGE"
    plot(input, output, num_columns_list, name, groups)

    print("Finish gender class process")

    groups = ["caucasian", "black"]

    input = "../results/glove/most_frequency_words/glove_race_WB_100000_most_frequency.csv"
    output = "../plots/glove/most_frequency_words/glove_race_WB_frequency_ratio.pdf"
    name = "GloVe"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/openAI/most_frequency_words/openAI_race_WB_100000_most_frequency.csv"
    output = "../plots/openAI/most_frequency_words/openai_race_WB_frequency_ratio.pdf"
    name = "OpenAI"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/fasttext/most_frequency_words/ft_race_WB_100000_most_frequency.csv"
    output = "../plots/fasttext/most_frequency_words/ft_race_WB_frequency_ratio.pdf"
    name = "FastText"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/cohere/most_frequency_words/cohere_race_WB_100000_most_frequency.csv"
    output = "../plots/cohere/most_frequency_words/cohere_race_WB_frequency_ratio.pdf"
    name = "Cohere"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/google/most_frequency_words/google_race_WB_100000_most_frequency.csv"
    output = "../plots/google/most_frequency_words/google_race_WB_frequency_ratio.pdf"
    name = "Google"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/microsoft/most_frequency_words/microsoft_race_WB_100000_most_frequency.csv"
    output = "../plots/microsoft/most_frequency_words/microsoft_race_WB_frequency_ratio.pdf"
    name = "Microsoft"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/BGE/most_frequency_words/BGE_race_WB_100000_most_frequency.csv"
    output = "../plots/BGE/most_frequency_words/BGE_race_WB_frequency_ratio.pdf"
    name = "BGE"
    plot(input, output, num_columns_list, name, groups)

    print("Finish first race class process")

    groups = ["caucasian", "asian"]

    input = "../results/glove/most_frequency_words/glove_race_WA_100000_most_frequency.csv"
    output = "../plots/glove/most_frequency_words/glove_race_WA_frequency_ratio.pdf"
    name = "GloVe"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/openAI/most_frequency_words/openAI_race_WA_100000_most_frequency.csv"
    output = "../plots/openAI/most_frequency_words/openai_race_WA_frequency_ratio.pdf"
    name = "OpenAI"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/fasttext/most_frequency_words/ft_race_WA_100000_most_frequency.csv"
    output = "../plots/fasttext/most_frequency_words/ft_race_WA_frequency_ratio.pdf"
    name = "FastText"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/cohere/most_frequency_words/cohere_race_WA_100000_most_frequency.csv"
    output = "../plots/cohere/most_frequency_words/cohere_race_WA_frequency_ratio.pdf"
    name = "Cohere"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/google/most_frequency_words/google_race_WA_100000_most_frequency.csv"
    output = "../plots/google/most_frequency_words/google_race_WA_frequency_ratio.pdf"
    name = "Google"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/microsoft/most_frequency_words/microsoft_race_WA_100000_most_frequency.csv"
    output = "../plots/microsoft/most_frequency_words/microsoft_race_WA_frequency_ratio.pdf"
    name = "Microsoft"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/BGE/most_frequency_words/BGE_race_WA_100000_most_frequency.csv"
    output = "../plots/BGE/most_frequency_words/BGE_race_WA_frequency_ratio.pdf"
    name = "BGE"
    plot(input, output, num_columns_list, name, groups)

    print("Finish second race class process")

    groups = ["asian", "black"]

    input = "../results/glove/most_frequency_words/glove_race_AB_100000_most_frequency.csv"
    output = "../plots/glove/most_frequency_words/glove_race_AB_frequency_ratio.pdf"
    name = "GloVe"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/openAI/most_frequency_words/openAI_race_AB_100000_most_frequency.csv"
    output = "../plots/openAI/most_frequency_words/openai_race_AB_frequency_ratio.pdf"
    name = "OpenAI"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/fasttext/most_frequency_words/ft_race_AB_100000_most_frequency.csv"
    output = "../plots/fasttext/most_frequency_words/ft_race_AB_frequency_ratio.pdf"
    name = "FastText"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/cohere/most_frequency_words/cohere_race_AB_100000_most_frequency.csv"
    output = "../plots/cohere/most_frequency_words/cohere_race_AB_frequency_ratio.pdf"
    name = "Cohere"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/google/most_frequency_words/google_race_AB_100000_most_frequency.csv"
    output = "../plots/google/most_frequency_words/google_race_AB_frequency_ratio.pdf"
    name = "Google"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/microsoft/most_frequency_words/microsoft_race_AB_100000_most_frequency.csv"
    output = "../plots/microsoft/most_frequency_words/microsoft_race_AB_frequency_ratio.pdf"
    name = "Microsoft"
    plot(input, output, num_columns_list, name, groups)

    input = "../results/BGE/most_frequency_words/BGE_race_AB_100000_most_frequency.csv"
    output = "../plots/BGE/most_frequency_words/BGE_race_AB_frequency_ratio.pdf"
    name = "BGE"
    plot(input, output, num_columns_list, name, groups)

    print("Finish third race class process")
