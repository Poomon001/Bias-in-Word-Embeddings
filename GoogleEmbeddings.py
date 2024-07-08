import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from dotenv import load_dotenv
import os
import time

''' NEED TO LOG IN WITH GOOGLE CLOUD SDK '''
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID_GOOGLE")
REGION = "us-central1"

MODEL = "text-embedding-004"
TASK = "SEMANTIC_SIMILARITY"
DIMENSION = 300

vertexai.init(project=PROJECT_ID, location=REGION)

# get word list
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
            input_file = os.path.join(input_dir, f"google_chunk_skip_{i * 250}.txt")
            if os.path.exists(input_file):
                with open(input_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
            else:
                print(f"File {input_file} does not exist.")

if __name__ == "__main__":
    STEP = 250
    word_list = get_word_list()

    for i in range(0, len(word_list), STEP):
        all_results = []
        chunk = word_list[i:i + STEP]

        """Embeds texts with a pre-trained, foundational model."""
        model = TextEmbeddingModel.from_pretrained(MODEL)
        inputs = [TextEmbeddingInput(text, TASK) for text in chunk]
        kwargs = dict(output_dimensionality=DIMENSION) if DIMENSION else {}
        embeddings = model.get_embeddings(inputs, **kwargs)
        [all_results.append(embedding.values) for embedding in embeddings]

        print(f"completed: ", i)

        writeTo(f"temp/google_embeddings/google_chunk_skip_{i}.txt", all_results, chunk)

        time.sleep(15)
        print("Resumed after 15 sec to avoid the request limitation")

    # Write all results to a file once all chunks are processed
    input_directory = "temp/google_embeddings"
    output_filename = "D:/Honour_Thesis_Data/google/google_100000_most_freq_skip.txt"
    num_files = 400
    concatenateFiles(input_directory, output_filename, num_files)

