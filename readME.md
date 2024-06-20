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

## 5. Product any *1000_most_*.ext files
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

## 5. Microsoft
Model: E5-large-v2

Document: https://github.com/microsoft/unilm/tree/master/e5