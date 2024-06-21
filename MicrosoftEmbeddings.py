from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model = AutoModel.from_pretrained('intfloat/e5-large-v2')

# get word list
WORDS_FILE = "openAI/glove_english_word_100000_most_freq_skip.txt"
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
            embedding_str = " ".join(str(vector) for vector in vectors)
            file.write(embedding_str + "\n")

if __name__ == "__main__":
    STEP = 5000
    word_list = get_word_list()
    all_results = []
    all_norm_results = []

    for i in range(0, len(word_list), STEP):
        chunk = word_list[i:i + STEP]

        chunk = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)

        # Get the embeddings
        with torch.no_grad():
            outputs = model(**chunk)
            embeddings = outputs.last_hidden_state

        word_embeddings = embeddings.mean(dim=1)

        normalized_embeddings = F.normalize(word_embeddings[:], p=2, dim=1)
        all_norm_results.extend(normalized_embeddings)

        word_embeddings = [word_embedding.tolist() for word_embedding in word_embeddings]
        all_results.extend(word_embeddings)

        print(f"completed: ", i)

    # without normalization
    writeTo("microsoft/microsoft_100000_most_freq_skip.txt", all_results)

    # with normalization
    writeTo("microsoft_norm/microsoft_norm_100000_most_freq_skip.txt", [all_norm_result.tolist() for all_norm_result in all_norm_results])