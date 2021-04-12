import re
from pathlib import Path
from typing import Sequence, Union, Set
import numpy as np

from .util import sentencepiece_load, http_get, load_word2vec_file
from .available_languages import wikicode, to_wikicode


class BPEmb():
    """
    A BPEmb model and utility functions for interacting with it.

    # Examples

    Load a BPEmb model for English:
    >>> bpemb_en = BPEmb(lang="en")

    Load a BPEmb model for Chinese and choose the vocabulary size (vs),
    that is, the number of byte-pair symbols:
    >>> bpemb_zh = BPEmb(lang="zh", vs=100000)

    Choose the embedding dimension:
    >>> bpemb_es = BPEmb(lang="es", vs=50000, dim=300)

    Byte-pair encode text:
    >>> bpemb_en.encode("stratford")
    ['▁strat', 'ford']

    >>> bpemb_en.encode("This is anarchism")
    ['▁this', '▁is', '▁an', 'arch', 'ism']

    >>> bpemb_zh.encode("这是一个中文句子")
    ['▁这是一个', '中文', '句子']

    Byte-pair encode text into IDs for performing an embedding lookup:
    >>> ids = bpemb_zh.encode_ids("这是一个中文句子")
    >>> ids
    [25950, 695, 20199]
    >>> bpemb_zh.vectors.shape
    (100000, 100)
    >>> embedded = bpemb_zh.vectors[ids]
    >>> embedded.shape
    (3, 100)

    Byte-pair encode and embed text:
    >>> bpemb_es.embed("No entendemos por qué.").shape
    (6, 300,)

    Decode byte-pair-encoded text:
    >>> bpemb_en.decode(['▁this', '▁is', '▁an', 'arch', 'ism'])
    'this is anarchism'

    The encode-decode roundtrip is lossy:
    >>> bpemb_en.decode(bpemb_en.encode("This is anarchism 101"))
    'this is anarchism 000'

    This is due to the preprocessing being applied before encoding:
    >>> bpemb_en.preprocess("This is anarchism 101")
    'this is anarchism 000'

    Decode byte-pair IDs:
    >>> bpemb_zh.decode_ids([25950, 695, 20199])
    '这是一个中文句子'


    Parameters
    ----------

    lang: ``str``, required
        Language of the byte-pair embeddings. The language string
        can be a:
            - Wikipedia edition code, e.g. ``en'' (recommended)
            - an ISO-639-3 language code, e.g. ``eng''
            - Wikipedia edition name, e.g. ``EgyptianArabic''
            - an ISO-639-3 language name, e.g. ``Egyptian Arabic''
        See:
        https://en.wikipedia.org/wiki/List_of_Wikipedias
        https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3_Name_Index.tab
    vs: ``ìnt'', optional (default = 10000)
        The vocabulary size of the byte pair model.
        This roughly, but not exactly, corresponds to the number of byte
        pair merge operations, since SentencePiece chooses the number of
        merges N depending on the number of unique characters C in the text
        to be encoded so that N + C = vs.
    dim: ``int'', optional (default = 100)
        The embedding dimensionality.
    cache_dir: ``Path'', optional (default = ``~/.cache/bpemb'')
        The folder in which downloaded BPEmb files will be cached.
    preprocess: ``bool'', optional (default = True)
        Whether to preprocess the text or not.
        Set to False if you have preprocessed the text already.
    encode_extra_options: ``str'' (default = None)
        Options that are directly passed to the SentencePiece encoder.
        See SentencePiece documentation for details.
    add_pad_emb: ``bool'', optional (default = False)
        Whether to add a special <pad> embedding to the byte pair
        embeddings, thereby increasing the vocabulary size to vs + 1.
        This embedding is initialized with zeros and appended to the end
        of the embedding matrix. Assuming "bpemb" is a BPEmb instance, the
        padding embedding can be looked up with "bpemb['<pad>']", or
        directly accessed with "bpemb.vectors[-1]".
    vs_fallback: ``bool'', optional (default = False)
        Vocabulary size fallback. Not all vocabulary sizes are available
        for all languages. For example, vs=1000 is not available for
        Chinese due to the large number of characters.
        When set to True, this option enables an automatic fallback to
        the closest available vocabulary size. For example,
        when selecting BPEmb("Chinese", vs=1000, vs_fallback=True),
        the actual vocabulary size would be 10000.
    segmentation_only: ``bool'', optional (default = False)
        If set to True, only the SentencePiece subword segmentation
        model will be loaded. Use this flag if you do not need the
        subword embeddings.
    model_file: ``Path'', optional (default = None)
        Path to a custom SentencePiece model file.
    emb_file: ``Path'', optional (default = None)
        Path to a custom embedding file. Supported formats are Word2Vec
        plain text and GenSim binary.
    """
    base_url = "https://nlp.h-its.org/bpemb/"
    emb_tpl = "{lang}/{lang}.wiki.bpe.vs{vs}.d{dim}.w2v.bin"
    model_tpl = "{lang}/{lang}.wiki.bpe.vs{vs}.model"
    archive_suffix = ".tar.gz"
    available_languages = wikicode

    def __init__(
            self,
            *,
            lang: str = None,
            vs: int = 10000,
            dim: int = 100,
            cache_dir: Path = Path.home() / Path(".cache/bpemb"),
            preprocess: bool = True,
            encode_extra_options: str = None,
            add_pad_emb: bool = False,
            vs_fallback: bool = True,
            segmentation_only: bool = False,
            model_file: Path = None,
            emb_file: Path = None):
        if lang is not None:
            self.lang = lang = BPEmb._get_lang(lang)
            if self.lang == 'multi':
                if dim != 300:
                    print('Setting dim=300 for multilingual BPEmb')
                    dim = 300
        else:
            assert (
                model_file is not None and
                emb_file is not None), (
                'Need to specify model_file and emb_file if no lang '
                'is given in BPEmb(...)')
            self.lang = lang
        if vs_fallback and lang:
            available = BPEmb.available_vocab_sizes(lang)
            if not available:
                raise ValueError("No BPEmb models for language " + lang)
            if vs not in available:
                available = sorted(available)
                _vs = vs
                if vs < available[0]:
                    vs = available[0]
                else:
                    vs = available[-1]
                print("BPEmb fallback: {} from vocab size {} to {}".format(
                    lang, _vs, vs))
        self.cache_dir = Path(cache_dir)
        if model_file is not None:
            # custom model file
            self.model_file = Path(model_file)
        else:
            model_file = self.model_tpl.format(lang=lang, vs=vs)
            self.model_file = self._load_file(model_file)
        self.spm = sentencepiece_load(self.model_file)
        self.vocab_size = self.vs = self.spm.get_piece_size()
        if encode_extra_options:
            self.spm.SetEncodeExtraOptions(encode_extra_options)
        self.do_preproc = preprocess
        self.BOS_str = "<s>"
        self.EOS_str = "</s>"
        self.BOS = self.spm.PieceToId(self.BOS_str)
        self.EOS = self.spm.PieceToId(self.EOS_str)
        self.segmentation_only = segmentation_only
        if not self.segmentation_only:
            if emb_file is not None:
                # custom embedding file
                self.emb_file = Path(emb_file)
            else:
                emb_file = self.emb_tpl.format(lang=lang, vs=vs, dim=dim)
                self.emb_file = self._load_file(emb_file, archive=True)
            self.emb = load_word2vec_file(self.emb_file, add_pad=add_pad_emb)
            self.most_similar = self.emb.most_similar
            self.dim = self.emb.vectors.shape[1]
            if dim is not None:
                assert self.dim == dim

    def __getitem__(self, key):
        return self.emb.__getitem__(key)

    @property
    def vectors(self):
        return self.emb.vectors

    @staticmethod
    def _get_lang(lang):
        if lang in {'multi', 'multilingual'}:
            return 'multi'
        if lang in wikicode:
            return lang
        try:
            return to_wikicode[lang]
        except:
            raise ValueError("Unknown language identifier: " + lang)

    def _load_file(self, file, archive=False, cache_dir=None):
        if not cache_dir:
            if hasattr(self, "cache_dir"):
                cache_dir = self.cache_dir
            else:
                from tempfile import mkdtemp
                cache_dir = mkdtemp()
        cached_file = Path(cache_dir) / file
        if cached_file.exists():
            return cached_file
        suffix = self.archive_suffix if archive else ""
        file_url = self.base_url + file + suffix
        print("downloading", file_url)
        return http_get(file_url, cached_file, ignore_tardir=True)

    def __repr__(self):
        if self.lang:
            lang_str = 'lang=' + self.lang
        else:
            lang_str = 'emb=' + self.emb_file.name
        return self.__class__.__name__ + \
            "({}, vs={}, dim={})".format(lang_str, self.vocab_size, self.dim)

    def encode(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair symbols.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(texts, self.spm.EncodeAsPieces)

    def encode_ids(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[int], Sequence[Sequence[int]]]:
        """Encode the supplied texts into byte-pair IDs.
        The byte-pair IDs correspond to row-indices into the embedding
        matrix.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(texts, self.spm.EncodeAsIds)

    def encode_with_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair symbols, adding
        an end-of-sentence symbol at the end of each encoded text.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: self.spm.EncodeAsPieces(t) + [self.EOS_str])

    def encode_ids_with_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[int], Sequence[Sequence[int]]]:
        """Encode the supplied texts into byte-pair IDs, adding
        an end-of-sentence symbol at the end of each encoded text.
        The byte-pair IDs correspond to row-indices into the embedding
        matrix.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: self.spm.EncodeAsIds(t) + [self.EOS])

    def encode_with_bos_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair symbols, adding
        a begin-of-sentence and an end-of-sentence symbol at the
        begin and end of each encoded text.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: (
                [self.BOS_str] + self.spm.EncodeAsPieces(t) + [self.EOS_str]))

    def encode_ids_with_bos_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[int], Sequence[Sequence[int]]]:
        """Encode the supplied texts into byte-pair IDs, adding
        a begin-of-sentence and an end-of-sentence symbol at the
        begin and end of each encoded text.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: [self.BOS] + self.spm.EncodeAsIds(t) + [self.EOS])

    def _encode(self, texts, fn):
        if isinstance(texts, str):
            if self.do_preproc:
                texts = self.preprocess(texts)
            return fn(texts)
        if self.do_preproc:
            texts = map(self.preprocess, texts)
        return list(map(fn, texts))

    def embed(self, text: str) -> np.ndarray:
        """Byte-pair encode text and return the corresponding byte-pair
        embeddings.

        Parameters
        ----------
        text: ``str'', required
            The text to encode and embed.

        Returns
        -------
        A matrix of shape (l, d), where l is the length of the byte-pair
        encoded text and d the embedding dimension.
        """
        ids = self.encode_ids(text)
        return self.emb.vectors[ids]

    def decode(
            self,
            pieces: Union[Sequence[str], Sequence[Sequence[str]]]
            ) -> Union[str, Sequence[str]]:
        """
        Decode the supplied byte-pair symbols.

        Parameters
        ----------
        pieces: ``Union[Sequence[str], Sequence[Sequence[str]]]'', required
            The byte-pair symbols to be decoded.

        Returns
        -------
            The decoded byte-pair symbols.
        """
        if isinstance(pieces[0], str):
            return self.spm.DecodePieces(pieces)
        return list(map(self.spm.DecodePieces, pieces))

    def decode_ids(
            self,
            ids: Union[Sequence[int], Sequence[Sequence[int]]]
            ) -> Union[str, Sequence[str]]:
        """
        Decode the supplied byte-pair IDs.

        Parameters
        ----------
        ids: ``Union[Sequence[int], Sequence[Sequence[int]]]'', required
            The byte-pair symbols to be decoded.

        Returns
        -------
            The decoded byte-pair IDs.
        """
        try:
            # try to decode list of lists
            return list(map(self.spm.DecodeIds, ids))
        except TypeError:
            try:
                # try to decode array
                return self.spm.DecodeIds(ids.tolist())
            except AttributeError:
                try:
                    # try to decode list of arrays
                    return list(map(self.spm.DecodeIds, ids.tolist()))
                except AttributeError:
                    # try to decode list
                    return self.spm.DecodeIds(ids)

    @staticmethod
    def preprocess(text: str) -> str:
        """
        Perform the preprocessing necessary for byte-pair encoding text
        one of BPEmb's pretrained BPE models.

        Parameters
        ----------
        text: ``str'', required
            The text to be preprocessed.

        Returns
        -------
        The preprocessed text.
        """
        return re.sub(r"\d", "0", text.lower())

    @property
    def pieces(self):
        try:
            return self.emb.index_to_key
        except AttributeError:
            return self.emb.index2word

    @property
    def words(self):
        return self.pieces

    @staticmethod
    def available_vocab_sizes(lang: str) -> Set[int]:
        """
        Return the available vocabulary sizes for the given language.

        Parameters
        ----------
        lang: ``str'', required
            The language identifier.

        Returns
        -------
            The available vocabulary sizes.
        """
        from .available_vocab_sizes import vocab_sizes
        lang = BPEmb._get_lang(lang)
        return vocab_sizes[lang]

    def __getstate__(self):
        state = self.__dict__.copy()
        # the SentencePiece instance is not serializable since it is a
        # SWIG object, so we need to delete it before serializing
        state['spm'] = None
        return state

    def __setstate__(self, state):
        # load SentencePiece after the BPEmb object has been unpickled
        model_file = (
            state["cache_dir"] / state["lang"] / state['model_file'].name)
        if not model_file.exists():
            model_rel_path = Path(state["lang"]) / model_file.name
            model_file = self._load_file(
                str(model_rel_path), cache_dir=state["cache_dir"])
        state['spm'] = sentencepiece_load(model_file)
        self.__dict__ = state


__all__ = [BPEmb]
