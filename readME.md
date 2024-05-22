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

Select the first 50k and the last 50k from crawl-300d-2M.vec

```bash
(head -n 50000 crawl-300d-2M.vec && tail -n 50000 crawl-300d-2M.vec)
```

Select the first 50k and the last 50k from glove.840B.300d.txt
```bash
(head -n 50000 glove.840B.300d.txt && tail -n 50000 glove.840B.300d.txt)
```