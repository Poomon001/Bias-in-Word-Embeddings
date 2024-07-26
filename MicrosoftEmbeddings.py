from transformers import AutoTokenizer, AutoModel
import torch
import csv

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model = AutoModel.from_pretrained('intfloat/e5-large-v2')

# get word list
WORDS_FILE = "raw/glove_english_word_100000_most_freq_skip.txt"
TOP_UNIVERSITIES_FILE = "concepts/top_50_universities.txt"
word_list = []

def get_word_list(word_file):
    lines = readFrom(word_file)
    return [word.strip() for word in lines]

def readFrom(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def writeTo(filename, response):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, vectors in zip(word_list, response):
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

if __name__ == "__main__":
    STEP = 5000
    word_list = get_word_list(WORDS_FILE)
    all_results = []

    for i in range(0, len(word_list), STEP):
        chunk = word_list[i:i + STEP]

        chunk = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)

        # Get the embeddings
        with torch.no_grad():
            outputs = model(**chunk)
            embeddings = outputs.last_hidden_state

        word_embeddings = embeddings.mean(dim=1)

        word_embeddings = [word_embedding.tolist() for word_embedding in word_embeddings]
        all_results.extend(word_embeddings)

        print(f"completed: ", i)

    writeTo("D:/Honour_Thesis_Data/microsoft/microsoft_100000_most_freq_skip.txt", all_results)

    print("Finish most_freq")

    word_list = get_word_list(TOP_UNIVERSITIES_FILE)
    all_results = []

    chunk = tokenizer(word_list, return_tensors='pt', padding=True, truncation=True)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**chunk)
        embeddings = outputs.last_hidden_state

    word_embeddings = embeddings.mean(dim=1)

    word_embeddings = [word_embedding.tolist() for word_embedding in word_embeddings]

    writeToCSV("D:/Honour_Thesis_Data/microsoft/microsoft_top_50_universities.csv", word_embeddings)

    print("Finish top university")