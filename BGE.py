from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3-unsupervised',  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

WORDS_FILE = "raw/glove_english_word_100000_most_freq_skip.txt"
RACE_WORDS_FILE = "raw/race_glove_english_word_100000_most_freq_skip.txt"
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

if __name__ == '__main__':
    STEP = 5000
    word_list = get_word_list()
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


