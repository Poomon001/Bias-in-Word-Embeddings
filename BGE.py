from FlagEmbedding import BGEM3FlagModel
import csv

model = BGEM3FlagModel('BAAI/bge-m3-unsupervised',  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

WORDS_FILE = "raw/glove_english_word_100000_most_freq_skip.txt"
TOP_UNIVERSITIES_FILE = "concepts/top_50_universities.txt"
word_list = []

def get_word_list(word_file):
    lines = readFrom(word_file)
    return [word.strip() for word in lines]

def readFrom(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def writeTo(filename, embeddings):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, vectors in zip(word_list, embeddings):
            file.write(word + " ")
            embedding_str = " ".join(str(vector) for vector in vectors)
            file.write(embedding_str + "\n")

def writeToCSV(filename, embeddings):
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        # Write header
        header = [0] + [f'{i}' for i in range(1, len(embeddings[0]) + 1)]
        writer.writerow(header)

        # Write each row
        for word, vector in zip(word_list, embeddings):
            vector_str = [str(v) for v in vector]
            writer.writerow([word] + vector_str)

if __name__ == '__main__':
    STEP = 5000
    word_list = get_word_list(WORDS_FILE)
    all_results = []

    for i in range(0, len(word_list), STEP):
        chunk = word_list[i:i + STEP]

        embeddings = model.encode(chunk,
                                  batch_size=12,
                                  max_length=512,
                                  )['dense_vecs']

        all_results.extend(embeddings)
        print(f"completed: ", i)

    writeTo("D:/Honour_Thesis_Data/BGE/BGE_100000_most_freq_skip.txt", all_results)

    print("Finish most_freq")

    word_list = get_word_list(TOP_UNIVERSITIES_FILE)

    embeddings = model.encode(word_list,
                              batch_size=12,
                              max_length=512,
                              )['dense_vecs']

    writeToCSV("D:/Honour_Thesis_Data/BGE/BGE_top_50_universities.csv", embeddings)

    print("Finish top university")
