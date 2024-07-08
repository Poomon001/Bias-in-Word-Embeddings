import os
from dotenv import load_dotenv
import cohere
import time

load_dotenv()
api_key = os.getenv("API_KEY_COHERE")
co = cohere.Client(api_key=api_key)


WORDS_FILE = "raw/glove_english_word_100000_most_freq_skip.txt"
word_list = []

def get_word_list():
    lines = readFrom(WORDS_FILE)
    return [word.strip() for word in lines]

def readFrom(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def writeTo(filename, response, chunk):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, vectors in zip(chunk, response):
            file.write(word + " ")
            embedding_str = " ".join(str(vector) for vector in vectors)
            file.write(embedding_str + "\n")

def concatenateFiles(input_dir, output_file, num_files):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i in range(num_files):
            input_file = os.path.join(input_dir, f"cohere_chunk_skip_{i * 2500}.txt")
            if os.path.exists(input_file):
                with open(input_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
            else:
                print(f"File {input_file} does not exist.")

if __name__ == '__main__':
    STEP = 2500
    model = "embed-english-v3.0"
    input_type = "clustering"
    word_list = get_word_list()
    all_results = []

    for i in range(22500, len(word_list), STEP):
        chunk = word_list[i:i + STEP]

        response = co.embed(
            model=model,
            texts=chunk,
            input_type=input_type,
            embedding_types=['float']
        ).embeddings

        print(f"completed: ", i)

        # all_results.extend(response.float_)

        writeTo(f"temp/cohere_embeddings/cohere_chunk_skip_{i}.txt", response.float_, chunk)

        time.sleep(65)
        print("Resumed after 65 sec to avoid the request limitation")


    # Write all results to a file once all chunks are processed
    input_directory = "temp/cohere_embeddings"
    output_filename = "D:/Honour_Thesis_Data/cohere/cohere_100000_most_freq_skip.txt"
    num_files = 40
    concatenateFiles(input_directory, output_filename, num_files)