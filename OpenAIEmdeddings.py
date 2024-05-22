import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
url = "https://api.openai.com/v1/embeddings"

WORDS_FILE = "raw/glove_word_100.txt"
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
    word_list = get_word_list()

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
        writeTo("openAI/openAI_100.txt", response.json()["data"])
    else:
        print("Error:", response.text)