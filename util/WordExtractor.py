'''
WordExtractor:
Extract only english words from word embedding files to a new txt file
'''

# Extract data from file
def readFrom(filename: str):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def writeTo(filename: str, lines: str):
    lines = content.split("\n")
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            if line:
                word = line.split()[0]
                # if word in word_list:
                file.write(word + "\n")

def writeToEmbeddings(filename: str, content: str):
    lines = content.split("\n")
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            if line:
                word = line.split()[0]
                # if word in word_list:
                file.write(line + "\n")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    content = readFrom("D:/Honour_Thesis_Data/raw/glove_100000_most_freq_skip.txt")
    writeTo("../raw/glove_english_word_100000_most_freq_skip.txt", content)



