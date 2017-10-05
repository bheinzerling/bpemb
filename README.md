# BPEmb

BPEmb is a collection of pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. Its intended use is as input for neural models in natural language processing.


## tl;dr

- Subwords allow guessing the meaning of unknown / out-of-vocabulary words. E.g., the suffix *-shire* in *Melfordshire* indicates a location.
- Byte-Pair Encoding gives a subword segmentation that is often good enough, without requiring tokenization or morphological analysis. In this case the BPE segmentation might be something like *melf ord shire*.
- Pre-trained byte-pair embeddings work surprisingly well, while requiring no tokenization and being much smaller than alternatives: the 11 MB BPEmb English model matches the results of the 6 GB FastText model in our evaluation.


## Example

Apply [BPE](https://github.com/rsennrich/subword-nmt) with 3000 merge operations, using [SentencePiece](https://github.com/google/sentencepiece):

```bash
$ echo melfordshire | spm_encode --model data/en/en.wiki.bpe.op3000.model
▁mel ford shire
```

Load an English BPEmb model with [gensim](https://github.com/RaRe-Technologies/gensim) and get BPE embedding vectors:

```Python
>>> from gensim.models import KeyedVectors
>>> model = KeyedVectors.load_word2vec_format("data/en/en.wiki.bpe.op3000.d100.w2v.bin", binary=True)
INFO:gensim.models.keyedvectors:loaded (3829, 100) matrix
>>> subwords = "▁mel ford shire".split()
>>> subwords
['▁mel', 'ford', 'shire']
>>> bpe_embs = model[subwords]
>>> bpe_embs.shape
(3, 100)
```

## Overview

#### What are subword embeddings and why should I use them?

If you are using word embeddings like word2vec or GloVe, you have probably encountered out-of-vocabulary words, i.e., words for which no embedding exists. A makeshift solution is to replace such words with an <unk> token and train a generic embedding representing such unknown words.

Subword approaches try to solve the unknown word problem differently, by assuming that you can reconstruct a word's meaning from its parts. For example, the suffix *-shire* lets you guess that *Melfordshire* is probably a location, or the suffix *-osis* that *Myxomatosis* might be a sickness.

There are many ways of splitting a word into subwords. A simple method is to split into characters and then learn to transform this character sequence into a vector representation by feeding it to a convolutional neural network (CNN) or a recurrent neural network (RNN), usually a long-short term memory (LSTM). This vector representation can then be used like a word embedding.

Another, more linguistically motivated way is a morphological analysis, but this requires tools and training data which might not be available for your language and domain of interest.

Enter Byte-Pair Encoding (BPE) [[Sennrich et al, 2016]](http://www.aclweb.org/anthology/P16-1162), an unsupervised subword segmentation method. BPE starts with a sequence of symbols, for example characters, and iteratively merges the most frequent symbol pair into a new symbol.

## Download BPEmb

Downloads for the 15 largest (by Wikipedia size) languages below. Downloads for all 275 languages [here](download.md).

| Language | Wikipedia edition | merge ops | model | vocab | 25 dims | 50 dims | 100 dims | 200 dims | 300 dims |
| - | - | - | - | - | - | - | - | - | - |
