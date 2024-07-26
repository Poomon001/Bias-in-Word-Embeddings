import requests
import os
from dotenv import load_dotenv
import csv

load_dotenv()
api_key = os.getenv("API_KEY_OPENAI")
url = "https://api.openai.com/v1/embeddings"

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
            embedding_str = " ".join(str(vector) for vector in vectors['embedding'])
            file.write(embedding_str + "\n")

def writeToCSV(filename, embeddings):
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        # Write header
        header = [0] + [f'{i}' for i in range(1, len(embeddings[0]['embedding']) + 1)]
        writer.writerow(header)

        # Write each row
        for word, vector in zip(word_list, embeddings):
            vector_str = [str(v) for v in vector['embedding']]
            writer.writerow([word] + vector_str)

if __name__ == '__main__':
    STEP = 2000
    word_list = get_word_list(WORDS_FILE)

    all_results = []

    for i in range(0, len(word_list), STEP):
        chunk = word_list[i:i + STEP]

        data = {
            "model": "text-embedding-3-small",
            "input": chunk
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Make the POST request
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            all_results.extend(response.json()["data"])
        else:
            print(f"Error with chunk {i // STEP + 1}: {response.text}")
            break

    # Write all results to a file once all chunks are processed
    writeTo("D:/Honour_Thesis_Data/openAI/openAI_100000_most_freq_skip.txt", all_results)

    print("Finish most_freq")

    word_list = get_word_list(TOP_UNIVERSITIES_FILE)

    data = {
        "model": "text-embedding-3-small",
        "input": word_list
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Make the POST request
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        writeToCSV("D:/Honour_Thesis_Data/openAI/openAI_top_50_universities.csv", response.json()["data"])
    else:
        print(f"Error with TOP_UNIVERSITIES: {response.text}")

    print("Finish top university")