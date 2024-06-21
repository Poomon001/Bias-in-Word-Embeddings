from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3-unsupervised',  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

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
            embedding_str = " ".join(str(vector) for vector in vectors['embedding'])
            file.write(embedding_str + "\n")

if __name__ == '__main__':
    STEP = 5
    word_list = get_word_list()[:11]

    all_results = []

    embeddings = model.encode(word_list,
                                batch_size=12,
                                max_length=1024,
                                )['dense_vecs']

print(embeddings)