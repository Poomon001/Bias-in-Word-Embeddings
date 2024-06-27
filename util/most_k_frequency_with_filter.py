import nltk
from nltk.corpus import stopwords
import string
import os
import re

# get a list of word to exclude
nltk.download('stopwords')
stop_word_list = stopwords.words('english')
punctuation_list = list(string.punctuation)
female_stimuli = ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]
male_stimuli = ["male", "he", "him", "his", "man", "boy", "son", "brother"]
skip_list = set(stop_word_list + punctuation_list + female_stimuli + male_stimuli)
number_pattern = re.compile(r'\d')  # Regex pattern to match any digit

def readFrom(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def is_valid_word(word):
    return word.isalpha() and len(word) > 2 and not number_pattern.search(word)

def process(inputFilename, outputFilename, k):
    total = 0
    content = readFrom(inputFilename)

    lines = content.split("\n")

    with open(outputFilename, 'w', encoding='utf-8') as file:
        # add female/male stimuli seperately
        for line in lines:
            if line:
                word = line.split()[0]
                if word in set(female_stimuli + male_stimuli):
                    total += 1
                    file.write(line + "\n")

        # add the rest excluding skip_words
        for line in lines:
            if line:
                word = line.split()[0]
                if word not in skip_list:
                    if is_valid_word(word):
                        if total < k:
                            file.write(line + "\n")
                            total += 1
                        else:
                            break

if __name__ == "__main__":
    print(skip_list)

    # filename = f"D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt"
    # process("../raw/glove_120k_most_freq.txt", filename, 100000)

    filename = f"D:/Honour_Thesis_Data/raw/ft_100000_most_freq_skip.csv"
    process("../raw/ft_120k_most_freq.vec", filename, 100000)
