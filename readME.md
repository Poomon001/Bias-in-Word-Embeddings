# Data

Download original data files

## 1. Download GloVe Word Embeddings

Download the full GloVe word embeddings file (`glove.840B.300d`) from the official GitHub repository.

- **GloVe Repository**: [StanfordNLP GloVe](https://github.com/stanfordnlp/GloVe)
- **Direct Download Link**: [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)

## 2. Download FastText Word Embeddings

Download the full FastText word embeddings file (`crawl-300d-2M.vec`) from the official FastText website.

- **FastText Vectors**: [FastText English Vectors](https://fasttext.cc/docs/en/english-vectors.html)
- **Direct Download Link**: [crawl-300d-2M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)

## 3. Download NRC-VAD Lexicon

Download the full NRC-VAD-Lexicon file (`NRC-VAD-Lexicon.txt`) from the official webpage of Saif Mohammad.

- **NRC-VAD Lexicon**: [NRC-VAD-Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html)
- **Direct Download Link**: [NRC-VAD-Lexicon.zip](https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip)# Bias-in-Word-Embeddings

## 4. Product ft_100k.csv and glove_100k.txt

Select the 120k most frequency from glove.840B.300d.txt

```bash
head -n 120000 glove.840B.300d.txt > glove_120k_most_freq.txt
```

## 5. Product any _1000*most*_.ext files

Example 1 .txt:

```bash
head -n 1000 glove_100000_most_freq_skip.txt > glove_1000_most_freq.txt
```

Example 2 .csv:
Need a header row = 1 + 1000 rows

```bash
head -n 1001 glove_100000_most_frequency.csv > glove_1000_most_frequency.csv
```

# Models

Testing models

## 1. GloVe

Document: https://nlp.stanford.edu/data/glove.840B.300d.zip

## 2. FastText

Document: https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

## 3. OpenAI

Model: text-embedding-3-small

Document: https://platform.openai.com/docs/guides/embeddings

## 4. Cohere

Model: embed-english-v3.0

Document: https://docs.cohere.com/reference/embed

## 5. Google

Model: text-embedding-004

Document: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api

requirement:

1. create Google Cloud account and create a new project
2. Download Google Cloud SDK
3. Authenticate:
   ```bash
   gcloud auth login
   ```
4. Set up credentials:
   ```bash
   gcloud auth application-default login
   ```

## 6. Microsoft

Model: E5-large-v2

Document: https://github.com/microsoft/unilm/tree/master/e5
Document: https://huggingface.co/intfloat/e5-large-v2

## 7. BGE (Beijing Academy of Artificial Intelligence)

Model: BAAI/bge-m3-unsupervised

Document: https://huggingface.co/BAAI/bge-m3
Document: https://huggingface.co/BAAI/bge-m3-unsupervised

#  Cluster Generation
## 1. clustering prompt: 

input the cluster of male/female words with the following prompt to ChatGPT 4 and Gemini
```bash
Cluster 0: Cache, Cherry, Chill, ...
Cluster 1: Alice, Amanda, Angela, ...
Cluster 2: Autumn, Cottage, Kitchen, ...
Cluster 3: Fashion, Flower, Frequency, ...
Cluster 4: Adventure, Emergency, Holiday, ...
Cluster 5: Artists, Dogs, Scholars, ...
Cluster 6: Computers, Families, Holidays, ...
Cluster 7: Beautiful, Daughter, Queen, ...
Cluster 8: Celebration, Evaluation, Upgrade, ...
Cluster 9: Dancing, Design, Dinner, ...
Cluster 10: Counseling, Happiness, Laser, ...

For each cluster of words, assign an appropriate unique concept such as Sports, Health and Relationships, Female Names, or Engineering and Electronics.
Output in json e.g {"0":title}
```

## Storage:

Store the result in a JSon for a later use
```bash
{
    "0": "General Descriptors and Objects",
    "1": "Female Names",
    "2": "Seasons, Home, and Food",
    "3": "Textiles and Household Items",
    "4": "Organizations, Events, and Technology",
    "5": "Various Nouns and Activities",
    "6": "Common Nouns and Activities",
    "7": "Female and Beauty-related Terms",
    "8": "Administrative and Enhancement Terms",
    "9": "Various Descriptors and Nouns",
    "10": "Assistance, Technology, and Household Items"
}
```

