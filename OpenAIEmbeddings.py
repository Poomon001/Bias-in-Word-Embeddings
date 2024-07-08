import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY_OPENAI")
url = "https://api.openai.com/v1/embeddings"

WORDS_FILE = "raw/glove_english_word_100000_most_freq_skip.txt"
word_list = []

def get_word_list():
    lines = readFrom(WORDS_FILE)
    return [word.strip() for word in lines]

def readFrom(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def writeTo(filename, response):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, vectors in zip(word_list, response):
            file.write(word + " ")
            embedding_str = " ".join(str(vector) for vector in vectors['embedding'])
            file.write(embedding_str + "\n")

if __name__ == '__main__':
    STEP = 2000
    word_list = get_word_list()

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