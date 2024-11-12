# Bias-in-Word-Embeddings
**GitHub**: [GitHub](https://github.com/Poomon001/Bias-in-Word-Embeddings)

## Data

Download original data files

### 1. Download GloVe Word Embeddings

Download the full GloVe word embeddings file (`glove.840B.300d`) from the official GitHub repository.

- **GloVe Repository**: [StanfordNLP GloVe](https://github.com/stanfordnlp/GloVe)
- **Direct Download Link**: [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)

### 2. Download FastText Word Embeddings

Download the full FastText word embeddings file (`crawl-300d-2M.vec`) from the official FastText website.

- **FastText Vectors**: [FastText English Vectors](https://fasttext.cc/docs/en/english-vectors.html)
- **Direct Download Link**: [crawl-300d-2M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)

### 3. Download NRC-VAD Lexicon

Download the full NRC-VAD-Lexicon file (`NRC-VAD-Lexicon.txt`) from the official webpage of Saif Mohammad.

- **NRC-VAD Lexicon**: [NRC-VAD-Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html)
- **Direct Download Link**: [NRC-VAD-Lexicon.zip](https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip)# Bias-in-Word-Embeddings

### 4. Product ft_100k.csv and glove_100k.txt

Select the 500k most frequency from glove.840B.300d.txt

```bash
head -n 500000 glove.840B.300d.txt > > glove_500k_most_freq.txt
```

## Models

Testing models

### 1. GloVe

Document: https://nlp.stanford.edu/data/glove.840B.300d.zip

### 2. FastText

Document: https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

### 3. OpenAI

Model: text-embedding-3-small

Document: https://platform.openai.com/docs/guides/embeddings

### 4. Cohere

Model: embed-english-v3.0

Document: https://docs.cohere.com/reference/embed

### 5. Google

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

### 6. Microsoft

Model: E5-large-v2

Document: https://github.com/microsoft/unilm/tree/master/e5
Document: https://huggingface.co/intfloat/e5-large-v2

### 7. BGE (Beijing Academy of Artificial Intelligence)

Model: BAAI/bge-m3-unsupervised

Document: https://huggingface.co/BAAI/bge-m3
Document: https://huggingface.co/BAAI/bge-m3-unsupervised

##  Cluster Generation
### 1. clustering prompt: Generate group_clusters_to_topics.json

Input the cluster of words from group1_over_group2_clusters_11.txt to ChatGPT 4 and Gemini with the following prompt:

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

## Stimuli Source
### Gender
based off the past research paper

### Race
Selected based on:

1. based off the GPT-4.0 genrator with the prompt
```bash
female_stimuli = ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]
male_stimuli = ["male", "he", "him", "his", "man", "boy", "son", "brother"]

If these are male and female stemuli. What should be stimuli for White, Black and Asian races?
```

2. Top highest population

## Concepts

### Big Tech
Pick big tech companies defining by "Mohamed Abdalla and Moustafa Abdalla. 2021. The Grey Hoodie Project: Big
tobacco, big tech, and the threat on academic integrity." paper

### Top University
Pick to 50 universities from Times Higher Education

Document: https://www.timeshighereducation.com/world-university-rankings/2024/world-ranking

