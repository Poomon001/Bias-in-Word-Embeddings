'''
WordExtractor:
Extract only english words from word embedding files to a new txt file
'''
word_list = [
    "cat", "dog", "house", "car", "tree", "computer", "book", "city", "river", "mountain",
    "beach", "sky", "sun", "moon", "star", "love", "hate", "happy", "sad", "laugh", "cry",
    "eat", "sleep", "run", "walk", "jump", "swim", "fly", "speak", "listen", "read", "write",
    "learn", "teach", "friend", "enemy", "family", "stranger", "food", "drink", "music", "art",
    "science", "history", "future", "past", "present", "dream", "reality", "imagination",
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "pineapple", "kiwi", "pear",
    "male", "he", "him", "his", "man", "boy", "son", "brother", "mango", "coconut",
    "pizza", "burger", "pasta", "rice", "sushi", "cake", "king", "cookie", "chocolate", "candy",
    "female", "she", "her", "hers", "woman", "girl", "daughter", "sister", "bread", "butter",
    "coffee", "tea", "juice", "soda", "water", "milk", "wine", "beer", "whiskey", "vodka", "queen",
    'Google', 'Amazon', 'Facebook', 'Microsoft', 'Apple', 'Nvidia', 'Intel', 'IBM', 'Huawei', 'Samsung',
    'Uber', 'Alibaba', 'GOOGLE', 'AMAZON', 'FACEBOOK', 'MICROSOFT', 'APPLE', 'NVIDIA', 'INTEL', 'IBM',
    'HUAWEI', 'SAMSUNG', 'UBER', 'ALIBABA'
]

# Extract data from file
def readFrom(filename: str):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def writeTo(filename: str, content: str):
    lines = content.split("\n")
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            if line:
                word = line.split()[0]
                # if word in word_list:
                file.write(word + "\n")

def writeToEmbeddings(filename: str, content: str):
    lines = content.split("\n")
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            if line:
                word = line.split()[0]
                # if word in word_list:
                file.write(line + "\n")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    content = readFrom("../raw/glove_100000_most_freq_skip.txt")
    writeTo("../openAI/glove_english_word_100000_most_freq_skip.txt", content)


