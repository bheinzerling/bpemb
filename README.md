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

## Download BPEmb

[Here](download.md)
