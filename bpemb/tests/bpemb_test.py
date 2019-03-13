import unittest
from ..bpemb import BPEmb


class BPEmbTest(unittest.TestCase):

    def test_sentencepiece_import(self):
        try:
            from sentencepiece import SentencePieceProcessor  # NOQA
        except ImportError:
            self.fail(
                "Could not import SentencePieceProcessor from sentencepiece.")

    def test_gensim_import(self):
        try:
            from gensim.models import KeyedVectors  # NOQA
        except ImportError:
            self.fail("Could not import KeyedVectors from gensim.models")

    def test_emb_model_download(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        self.assertTrue(bpemb.emb_file.exists())
        self.assertTrue(bpemb.model_file.exists())

    def test_emb_dim_25(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        self.assertEqual(25, bpemb.emb.vectors.shape[1])

    def test_emb_dim_50(self):
        bpemb = BPEmb(lang="fr", vs=1000, dim=50)
        self.assertEqual(50, bpemb.emb.vectors.shape[1])

    def test_emb_dim_100(self):
        bpemb = BPEmb(lang="de", vs=1000, dim=100)
        self.assertEqual(100, bpemb.emb.vectors.shape[1])

    def test_emb_vs_1000(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        self.assertEqual(1000, bpemb.emb.vectors.shape[0])

    def test_emb_vs_5000(self):
        bpemb = BPEmb(lang="fr", vs=5000, dim=25)
        self.assertEqual(5000, bpemb.emb.vectors.shape[0])

    def test_emb_vs_10000(self):
        bpemb = BPEmb(lang="de", vs=10000, dim=25)
        self.assertEqual(10000, bpemb.emb.vectors.shape[0])

    def test_encode_decode_roundtrip(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        text = "this is a test"
        self.assertEqual(text, bpemb.decode(bpemb.encode(text)))

    def test_encode_ids_decode_ids_roundtrip(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        text = "this is a test"
        self.assertEqual(text, bpemb.decode_ids(bpemb.encode_ids(text)))

    def test_encode_decode_many_roundtrip(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        texts = ["this is a test", "another test", "and one more"]
        self.assertEqual(texts, bpemb.decode(bpemb.encode(texts)))

    def test_encode_ids_decode_ids_many_roundtrip(self):
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        texts = ["this is a test", "another test", "and one more"]
        self.assertEqual(texts, bpemb.decode_ids(bpemb.encode_ids(texts)))

    def test_pickle(self):
        import pickle
        from tempfile import mkdtemp
        from pathlib import Path
        bpemb = BPEmb(lang="en", vs=1000, dim=25)
        pklfile = Path(mkdtemp()) / "bpemb.pkl"
        with pklfile.open("wb") as out:
            pickle.dump(bpemb, out)
        with pklfile.open("rb") as pkl:
            bpemb_unpickled = pickle.load(pkl)
        text = "this is a test"
        self.assertEqual(
            text, bpemb.decode_ids(bpemb_unpickled.encode_ids(text)))
        self.assertEqual(
            text, bpemb_unpickled.decode_ids(bpemb.encode_ids(text)))

    def test_pickle_custom_cache_dir(self):
        import pickle
        from tempfile import mkdtemp
        from pathlib import Path
        bpemb = BPEmb(lang="en", vs=1000, dim=25, cache_dir=mkdtemp())
        pklfile = Path(mkdtemp()) / "bpemb.pkl"
        with pklfile.open("wb") as out:
            pickle.dump(bpemb, out)
        with pklfile.open("rb") as pkl:
            bpemb_unpickled = pickle.load(pkl)
        text = "this is a test"
        self.assertEqual(
            text, bpemb.decode_ids(bpemb_unpickled.encode_ids(text)))
        self.assertEqual(
            text, bpemb_unpickled.decode_ids(bpemb.encode_ids(text)))

    def test_pickle_custom_no_cache(self):
        import pickle
        from tempfile import mkdtemp
        from pathlib import Path
        import shutil
        cache_dir = mkdtemp()
        bpemb = BPEmb(lang="en", vs=1000, dim=25, cache_dir=cache_dir)
        shutil.rmtree(cache_dir)
        pklfile = Path(mkdtemp()) / "bpemb.pkl"
        with pklfile.open("wb") as out:
            pickle.dump(bpemb, out)
        with pklfile.open("rb") as pkl:
            bpemb_unpickled = pickle.load(pkl)
        text = "this is a test"
        self.assertEqual(
            text, bpemb.decode_ids(bpemb_unpickled.encode_ids(text)))
        self.assertEqual(
            text, bpemb_unpickled.decode_ids(bpemb.encode_ids(text)))

    def test_pad_lookup(self):
        dim = 25
        bpemb = BPEmb(lang="en", vs=1000, dim=dim, add_pad_emb=True)
        assert set(bpemb["<pad>"].tolist()) == {0}

    def test_pad_index(self):
        dim = 25
        bpemb = BPEmb(lang="en", vs=1000, dim=dim, add_pad_emb=True)
        assert set(bpemb.vectors[-1]) == {0}


if __name__ == "__main__":
    unittest.main()
