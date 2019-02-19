# BPEmb

BPEmb is a collection of pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. Its intended use is as input for neural models in natural language processing.

[Website](https://nlp.h-its.org/bpemb) ・ 
[Usage](#usage) ・ 
[Download](#downloads-for-each-language) ・ 
[Paper (pdf)](http://www.lrec-conf.org/proceedings/lrec2018/pdf/1049.pdf) ・ 
[Citing BPEmb](#citing-bpemb)



## Usage

Install BPEmb with pip:

```bash
pip install bpemb
```

Embeddings and SentencePiece models will be downloaded automatically the first time you use them.

```python
>>> from bpemb import BPEmb
# load English BPEmb model with default vocabulary size (10k) and 50-dimensional embeddings
>>> bpemb_en = BPEmb(lang="en", dim=50)
downloading https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.model
downloading https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.d50.w2v.bin.tar.gz
```

You can do two main things with BPEmb. The first is subword segmentation:
```python
# apply English BPE subword segmentation model
>>> bpemb_en.encode("Stratford")
['▁strat', 'ford']
# load Chinese BPEmb model with vocabulary size 100k and default (100-dim) embeddings
>>> bpemb_zh = BPEmb(lang="zh", vs=100000)
# apply Chinese BPE subword segmentation model
>>> bpemb_zh.encode("这是一个中文句子")  # "This is a Chinese sentence."
['▁这是一个', '中文', '句子']  # ["This is a", "Chinese", "sentence"]
```

If / how a word gets split depends on the vocabulary size. Generally, a smaller vocabulary size will yield a segmentation into many subwords, while a large vocabulary size will result in frequent words not being split:

| vocabulary size | segmentation |
| --- | --- |
| 1000 | ['▁str', 'at', 'f', 'ord'] |
| 3000 |  ['▁str', 'at', 'ford'] |
| 5000 | ['▁str', 'at', 'ford'] |
| 10000 | ['▁strat', 'ford'] |
| 25000 | ['▁stratford'] |
| 50000 | ['▁stratford'] |
| 100000 | ['▁stratford'] |
| 200000 | ['▁stratford'] |


The second purpose of BPEmb is to provide pretrained subword embeddings:

```python
# Embeddings are wrapped in a gensim KeyedVectors object
>>> type(bpemb_zh.emb)
gensim.models.keyedvectors.Word2VecKeyedVectors
# You can use BPEmb objects like gensim KeyedVectors
>>> bpemb_en.most_similar("ford")
[('bury', 0.8745079040527344),
 ('ton', 0.8725000619888306),
 ('well', 0.871537446975708),
 ('ston', 0.8701574206352234),
 ('worth', 0.8672043085098267),
 ('field', 0.859795331954956),
 ('ley', 0.8591548204421997),
 ('ington', 0.8126075267791748),
 ('bridge', 0.8099068999290466),
 ('brook', 0.7979353070259094)]
>>> type(bpemb_en.vectors)
numpy.ndarray
>>> bpemb_en.vectors.shape
(10000, 50)
>>> bpemb_zh.vectors.shape
(100000, 100)
```

To use subword embeddings in your neural network, either encode your input into subword IDs:
```python
>>> ids = bpemb_zh.encode_ids("这是一个中文句子")
[25950, 695, 20199]
>>> bpemb_zh.vectors[ids].shape
(3, 100)
```

Or use the `embed` method:
```python
# apply Chinese subword segmentation and perform embedding lookup
>>> bpemb_zh.embed("这是一个中文句子").shape
(3, 100)
```

# Downloads for each language

[ab (Abkhazian)](http://nlp.h-its.org/bpemb/ab) ・ 
[ace (Achinese)](http://nlp.h-its.org/bpemb/ace) ・ 
[ady (Adyghe)](http://nlp.h-its.org/bpemb/ady) ・ 
[af (Afrikaans)](http://nlp.h-its.org/bpemb/af) ・ 
[ak (Akan)](http://nlp.h-its.org/bpemb/ak) ・ 
[als (Alemannic)](http://nlp.h-its.org/bpemb/als) ・ 
[am (Amharic)](http://nlp.h-its.org/bpemb/am) ・ 
[an (Aragonese)](http://nlp.h-its.org/bpemb/an) ・ 
[ang (Old English)](http://nlp.h-its.org/bpemb/ang) ・ 
[ar (Arabic)](http://nlp.h-its.org/bpemb/ar) ・ 
[arc (Official Aramaic)](http://nlp.h-its.org/bpemb/arc) ・ 
[arz (Egyptian Arabic)](http://nlp.h-its.org/bpemb/arz) ・ 
[as (Assamese)](http://nlp.h-its.org/bpemb/as) ・ 
[ast (Asturian)](http://nlp.h-its.org/bpemb/ast) ・ 
[atj (Atikamekw)](http://nlp.h-its.org/bpemb/atj) ・ 
[av (Avaric)](http://nlp.h-its.org/bpemb/av) ・ 
[ay (Aymara)](http://nlp.h-its.org/bpemb/ay) ・ 
[az (Azerbaijani)](http://nlp.h-its.org/bpemb/az) ・ 
[azb (South Azerbaijani)](http://nlp.h-its.org/bpemb/azb)

[ba (Bashkir)](http://nlp.h-its.org/bpemb/ba) ・ 
[bar (Bavarian)](http://nlp.h-its.org/bpemb/bar) ・ 
[bcl (Central Bikol)](http://nlp.h-its.org/bpemb/bcl) ・ 
[be (Belarusian)](http://nlp.h-its.org/bpemb/be) ・ 
[bg (Bulgarian)](http://nlp.h-its.org/bpemb/bg) ・ 
[bi (Bislama)](http://nlp.h-its.org/bpemb/bi) ・ 
[bjn (Banjar)](http://nlp.h-its.org/bpemb/bjn) ・ 
[bm (Bambara)](http://nlp.h-its.org/bpemb/bm) ・ 
[bn (Bengali)](http://nlp.h-its.org/bpemb/bn) ・ 
[bo (Tibetan)](http://nlp.h-its.org/bpemb/bo) ・ 
[bpy (Bishnupriya)](http://nlp.h-its.org/bpemb/bpy) ・ 
[br (Breton)](http://nlp.h-its.org/bpemb/br) ・ 
[bs (Bosnian)](http://nlp.h-its.org/bpemb/bs) ・ 
[bug (Buginese)](http://nlp.h-its.org/bpemb/bug) ・ 
[bxr (Russia Buriat)](http://nlp.h-its.org/bpemb/bxr)

[ca (Catalan)](http://nlp.h-its.org/bpemb/ca) ・ 
[cdo (Min Dong Chinese)](http://nlp.h-its.org/bpemb/cdo) ・ 
[ce (Chechen)](http://nlp.h-its.org/bpemb/ce) ・ 
[ceb (Cebuano)](http://nlp.h-its.org/bpemb/ceb) ・ 
[ch (Chamorro)](http://nlp.h-its.org/bpemb/ch) ・ 
[chr (Cherokee)](http://nlp.h-its.org/bpemb/chr) ・ 
[chy (Cheyenne)](http://nlp.h-its.org/bpemb/chy) ・ 
[ckb (Central Kurdish)](http://nlp.h-its.org/bpemb/ckb) ・ 
[co (Corsican)](http://nlp.h-its.org/bpemb/co) ・ 
[cr (Cree)](http://nlp.h-its.org/bpemb/cr) ・ 
[crh (Crimean Tatar)](http://nlp.h-its.org/bpemb/crh) ・ 
[cs (Czech)](http://nlp.h-its.org/bpemb/cs) ・ 
[csb (Kashubian)](http://nlp.h-its.org/bpemb/csb) ・ 
[cu (Church Slavic)](http://nlp.h-its.org/bpemb/cu) ・ 
[cv (Chuvash)](http://nlp.h-its.org/bpemb/cv) ・ 
[cy (Welsh)](http://nlp.h-its.org/bpemb/cy)

[da (Danish)](http://nlp.h-its.org/bpemb/da) ・ 
[de (German)](http://nlp.h-its.org/bpemb/de) ・ 
[din (Dinka)](http://nlp.h-its.org/bpemb/din) ・ 
[diq (Dimli)](http://nlp.h-its.org/bpemb/diq) ・ 
[dsb (Lower Sorbian)](http://nlp.h-its.org/bpemb/dsb) ・ 
[dty (Dotyali)](http://nlp.h-its.org/bpemb/dty) ・ 
[dv (Dhivehi)](http://nlp.h-its.org/bpemb/dv) ・ 
[dz (Dzongkha)](http://nlp.h-its.org/bpemb/dz)

[ee (Ewe)](http://nlp.h-its.org/bpemb/ee) ・ 
[el (Modern Greek)](http://nlp.h-its.org/bpemb/el) ・ 
[en (English)](http://nlp.h-its.org/bpemb/en) ・ 
[eo (Esperanto)](http://nlp.h-its.org/bpemb/eo) ・ 
[es (Spanish)](http://nlp.h-its.org/bpemb/es) ・ 
[et (Estonian)](http://nlp.h-its.org/bpemb/et) ・ 
[eu (Basque)](http://nlp.h-its.org/bpemb/eu) ・ 
[ext (Extremaduran)](http://nlp.h-its.org/bpemb/ext)

[fa (Persian)](http://nlp.h-its.org/bpemb/fa) ・ 
[ff (Fulah)](http://nlp.h-its.org/bpemb/ff) ・ 
[fi (Finnish)](http://nlp.h-its.org/bpemb/fi) ・ 
[fj (Fijian)](http://nlp.h-its.org/bpemb/fj) ・ 
[fo (Faroese)](http://nlp.h-its.org/bpemb/fo) ・ 
[fr (French)](http://nlp.h-its.org/bpemb/fr) ・ 
[frp (Arpitan)](http://nlp.h-its.org/bpemb/frp) ・ 
[frr (Northern Frisian)](http://nlp.h-its.org/bpemb/frr) ・ 
[fur (Friulian)](http://nlp.h-its.org/bpemb/fur) ・ 
[fy (Western Frisian)](http://nlp.h-its.org/bpemb/fy)

[ga (Irish)](http://nlp.h-its.org/bpemb/ga) ・ 
[gag (Gagauz)](http://nlp.h-its.org/bpemb/gag) ・ 
[gan (Gan Chinese)](http://nlp.h-its.org/bpemb/gan) ・ 
[gd (Scottish Gaelic)](http://nlp.h-its.org/bpemb/gd) ・ 
[gl (Galician)](http://nlp.h-its.org/bpemb/gl) ・ 
[glk (Gilaki)](http://nlp.h-its.org/bpemb/glk) ・ 
[gn (Guarani)](http://nlp.h-its.org/bpemb/gn) ・ 
[gom (Goan Konkani)](http://nlp.h-its.org/bpemb/gom) ・ 
[got (Gothic)](http://nlp.h-its.org/bpemb/got) ・ 
[gu (Gujarati)](http://nlp.h-its.org/bpemb/gu) ・ 
[gv (Manx)](http://nlp.h-its.org/bpemb/gv)

[ha (Hausa)](http://nlp.h-its.org/bpemb/ha) ・ 
[hak (Hakka Chinese)](http://nlp.h-its.org/bpemb/hak) ・ 
[haw (Hawaiian)](http://nlp.h-its.org/bpemb/haw) ・ 
[he (Hebrew)](http://nlp.h-its.org/bpemb/he) ・ 
[hi (Hindi)](http://nlp.h-its.org/bpemb/hi) ・ 
[hif (Fiji Hindi)](http://nlp.h-its.org/bpemb/hif) ・ 
[hr (Croatian)](http://nlp.h-its.org/bpemb/hr) ・ 
[hsb (Upper Sorbian)](http://nlp.h-its.org/bpemb/hsb) ・ 
[ht (Haitian)](http://nlp.h-its.org/bpemb/ht) ・ 
[hu (Hungarian)](http://nlp.h-its.org/bpemb/hu) ・ 
[hy (Armenian)](http://nlp.h-its.org/bpemb/hy)

[ia (Interlingua)](http://nlp.h-its.org/bpemb/ia) ・ 
[id (Indonesian)](http://nlp.h-its.org/bpemb/id) ・ 
[ie (Interlingue)](http://nlp.h-its.org/bpemb/ie) ・ 
[ig (Igbo)](http://nlp.h-its.org/bpemb/ig) ・ 
[ik (Inupiaq)](http://nlp.h-its.org/bpemb/ik) ・ 
[ilo (Iloko)](http://nlp.h-its.org/bpemb/ilo) ・ 
[io (Ido)](http://nlp.h-its.org/bpemb/io) ・ 
[is (Icelandic)](http://nlp.h-its.org/bpemb/is) ・ 
[it (Italian)](http://nlp.h-its.org/bpemb/it) ・ 
[iu (Inuktitut)](http://nlp.h-its.org/bpemb/iu)

[ja (Japanese)](http://nlp.h-its.org/bpemb/ja) ・ 
[jam (Jamaican Creole English)](http://nlp.h-its.org/bpemb/jam) ・ 
[jbo (Lojban)](http://nlp.h-its.org/bpemb/jbo) ・ 
[jv (Javanese)](http://nlp.h-its.org/bpemb/jv)

[ka (Georgian)](http://nlp.h-its.org/bpemb/ka) ・ 
[kaa (Kara-Kalpak)](http://nlp.h-its.org/bpemb/kaa) ・ 
[kab (Kabyle)](http://nlp.h-its.org/bpemb/kab) ・ 
[kbd (Kabardian)](http://nlp.h-its.org/bpemb/kbd) ・ 
[kbp (Kabiyè)](http://nlp.h-its.org/bpemb/kbp) ・ 
[kg (Kongo)](http://nlp.h-its.org/bpemb/kg) ・ 
[ki (Kikuyu)](http://nlp.h-its.org/bpemb/ki) ・ 
[kk (Kazakh)](http://nlp.h-its.org/bpemb/kk) ・ 
[kl (Kalaallisut)](http://nlp.h-its.org/bpemb/kl) ・ 
[km (Central Khmer)](http://nlp.h-its.org/bpemb/km) ・ 
[kn (Kannada)](http://nlp.h-its.org/bpemb/kn) ・ 
[ko (Korean)](http://nlp.h-its.org/bpemb/ko) ・ 
[koi (Komi-Permyak)](http://nlp.h-its.org/bpemb/koi) ・ 
[krc (Karachay-Balkar)](http://nlp.h-its.org/bpemb/krc) ・ 
[ks (Kashmiri)](http://nlp.h-its.org/bpemb/ks) ・ 
[ksh (Kölsch)](http://nlp.h-its.org/bpemb/ksh) ・ 
[ku (Kurdish)](http://nlp.h-its.org/bpemb/ku) ・ 
[kv (Komi)](http://nlp.h-its.org/bpemb/kv) ・ 
[kw (Cornish)](http://nlp.h-its.org/bpemb/kw) ・ 
[ky (Kirghiz)](http://nlp.h-its.org/bpemb/ky)

[la (Latin)](http://nlp.h-its.org/bpemb/la) ・ 
[lad (Ladino)](http://nlp.h-its.org/bpemb/lad) ・ 
[lb (Luxembourgish)](http://nlp.h-its.org/bpemb/lb) ・ 
[lbe (Lak)](http://nlp.h-its.org/bpemb/lbe) ・ 
[lez (Lezghian)](http://nlp.h-its.org/bpemb/lez) ・ 
[lg (Ganda)](http://nlp.h-its.org/bpemb/lg) ・ 
[li (Limburgan)](http://nlp.h-its.org/bpemb/li) ・ 
[lij (Ligurian)](http://nlp.h-its.org/bpemb/lij) ・ 
[lmo (Lombard)](http://nlp.h-its.org/bpemb/lmo) ・ 
[ln (Lingala)](http://nlp.h-its.org/bpemb/ln) ・ 
[lo (Lao)](http://nlp.h-its.org/bpemb/lo) ・ 
[lrc (Northern Luri)](http://nlp.h-its.org/bpemb/lrc) ・ 
[lt (Lithuanian)](http://nlp.h-its.org/bpemb/lt) ・ 
[ltg (Latgalian)](http://nlp.h-its.org/bpemb/ltg) ・ 
[lv (Latvian)](http://nlp.h-its.org/bpemb/lv)

[mai (Maithili)](http://nlp.h-its.org/bpemb/mai) ・ 
[mdf (Moksha)](http://nlp.h-its.org/bpemb/mdf) ・ 
[mg (Malagasy)](http://nlp.h-its.org/bpemb/mg) ・ 
[mh (Marshallese)](http://nlp.h-its.org/bpemb/mh) ・ 
[mhr (Eastern Mari)](http://nlp.h-its.org/bpemb/mhr) ・ 
[mi (Maori)](http://nlp.h-its.org/bpemb/mi) ・ 
[min (Minangkabau)](http://nlp.h-its.org/bpemb/min) ・ 
[mk (Macedonian)](http://nlp.h-its.org/bpemb/mk) ・ 
[ml (Malayalam)](http://nlp.h-its.org/bpemb/ml) ・ 
[mn (Mongolian)](http://nlp.h-its.org/bpemb/mn) ・ 
[mr (Marathi)](http://nlp.h-its.org/bpemb/mr) ・ 
[mrj (Western Mari)](http://nlp.h-its.org/bpemb/mrj) ・ 
[ms (Malay)](http://nlp.h-its.org/bpemb/ms) ・ 
[mt (Maltese)](http://nlp.h-its.org/bpemb/mt) ・ 
[mwl (Mirandese)](http://nlp.h-its.org/bpemb/mwl) ・ 
[my (Burmese)](http://nlp.h-its.org/bpemb/my) ・ 
[myv (Erzya)](http://nlp.h-its.org/bpemb/myv) ・ 
[mzn (Mazanderani)](http://nlp.h-its.org/bpemb/mzn)

[na (Nauru)](http://nlp.h-its.org/bpemb/na) ・ 
[nap (Neapolitan)](http://nlp.h-its.org/bpemb/nap) ・ 
[nds (Low German)](http://nlp.h-its.org/bpemb/nds) ・ 
[ne (Nepali)](http://nlp.h-its.org/bpemb/ne) ・ 
[new (Newari)](http://nlp.h-its.org/bpemb/new) ・ 
[ng (Ndonga)](http://nlp.h-its.org/bpemb/ng) ・ 
[nl (Dutch)](http://nlp.h-its.org/bpemb/nl) ・ 
[nn (Norwegian Nynorsk)](http://nlp.h-its.org/bpemb/nn) ・ 
[no (Norwegian)](http://nlp.h-its.org/bpemb/no) ・ 
[nov (Novial)](http://nlp.h-its.org/bpemb/nov) ・ 
[nrm (Narom)](http://nlp.h-its.org/bpemb/nrm) ・ 
[nso (Pedi)](http://nlp.h-its.org/bpemb/nso) ・ 
[nv (Navajo)](http://nlp.h-its.org/bpemb/nv) ・ 
[ny (Nyanja)](http://nlp.h-its.org/bpemb/ny)

[oc (Occitan)](http://nlp.h-its.org/bpemb/oc) ・ 
[olo (Livvi)](http://nlp.h-its.org/bpemb/olo) ・ 
[om (Oromo)](http://nlp.h-its.org/bpemb/om) ・ 
[or (Oriya)](http://nlp.h-its.org/bpemb/or) ・ 
[os (Ossetian)](http://nlp.h-its.org/bpemb/os)

[pa (Panjabi)](http://nlp.h-its.org/bpemb/pa) ・ 
[pag (Pangasinan)](http://nlp.h-its.org/bpemb/pag) ・ 
[pam (Pampanga)](http://nlp.h-its.org/bpemb/pam) ・ 
[pap (Papiamento)](http://nlp.h-its.org/bpemb/pap) ・ 
[pcd (Picard)](http://nlp.h-its.org/bpemb/pcd) ・ 
[pdc (Pennsylvania German)](http://nlp.h-its.org/bpemb/pdc) ・ 
[pfl (Pfaelzisch)](http://nlp.h-its.org/bpemb/pfl) ・ 
[pi (Pali)](http://nlp.h-its.org/bpemb/pi) ・ 
[pih (Pitcairn-Norfolk)](http://nlp.h-its.org/bpemb/pih) ・ 
[pl (Polish)](http://nlp.h-its.org/bpemb/pl) ・ 
[pms (Piemontese)](http://nlp.h-its.org/bpemb/pms) ・ 
[pnb (Western Panjabi)](http://nlp.h-its.org/bpemb/pnb) ・ 
[pnt (Pontic)](http://nlp.h-its.org/bpemb/pnt) ・ 
[ps (Pushto)](http://nlp.h-its.org/bpemb/ps) ・ 
[pt (Portuguese)](http://nlp.h-its.org/bpemb/pt)

[qu (Quechua)](http://nlp.h-its.org/bpemb/qu)

[rm (Romansh)](http://nlp.h-its.org/bpemb/rm) ・ 
[rmy (Vlax Romani)](http://nlp.h-its.org/bpemb/rmy) ・ 
[rn (Rundi)](http://nlp.h-its.org/bpemb/rn) ・ 
[ro (Romanian)](http://nlp.h-its.org/bpemb/ro) ・ 
[ru (Russian)](http://nlp.h-its.org/bpemb/ru) ・ 
[rue (Rusyn)](http://nlp.h-its.org/bpemb/rue) ・ 
[rw (Kinyarwanda)](http://nlp.h-its.org/bpemb/rw)

[sa (Sanskrit)](http://nlp.h-its.org/bpemb/sa) ・ 
[sah (Yakut)](http://nlp.h-its.org/bpemb/sah) ・ 
[sc (Sardinian)](http://nlp.h-its.org/bpemb/sc) ・ 
[scn (Sicilian)](http://nlp.h-its.org/bpemb/scn) ・ 
[sco (Scots)](http://nlp.h-its.org/bpemb/sco) ・ 
[sd (Sindhi)](http://nlp.h-its.org/bpemb/sd) ・ 
[se (Northern Sami)](http://nlp.h-its.org/bpemb/se) ・ 
[sg (Sango)](http://nlp.h-its.org/bpemb/sg) ・ 
[sh (Serbo-Croatian)](http://nlp.h-its.org/bpemb/sh) ・ 
[si (Sinhala)](http://nlp.h-its.org/bpemb/si) ・ 
[sk (Slovak)](http://nlp.h-its.org/bpemb/sk) ・ 
[sl (Slovenian)](http://nlp.h-its.org/bpemb/sl) ・ 
[sm (Samoan)](http://nlp.h-its.org/bpemb/sm) ・ 
[sn (Shona)](http://nlp.h-its.org/bpemb/sn) ・ 
[so (Somali)](http://nlp.h-its.org/bpemb/so) ・ 
[sq (Albanian)](http://nlp.h-its.org/bpemb/sq) ・ 
[sr (Serbian)](http://nlp.h-its.org/bpemb/sr) ・ 
[srn (Sranan Tongo)](http://nlp.h-its.org/bpemb/srn) ・ 
[ss (Swati)](http://nlp.h-its.org/bpemb/ss) ・ 
[st (Southern Sotho)](http://nlp.h-its.org/bpemb/st) ・ 
[stq (Saterfriesisch)](http://nlp.h-its.org/bpemb/stq) ・ 
[su (Sundanese)](http://nlp.h-its.org/bpemb/su) ・ 
[sv (Swedish)](http://nlp.h-its.org/bpemb/sv) ・ 
[sw (Swahili)](http://nlp.h-its.org/bpemb/sw) ・ 
[szl (Silesian)](http://nlp.h-its.org/bpemb/szl)

[ta (Tamil)](http://nlp.h-its.org/bpemb/ta) ・ 
[tcy (Tulu)](http://nlp.h-its.org/bpemb/tcy) ・ 
[te (Telugu)](http://nlp.h-its.org/bpemb/te) ・ 
[tet (Tetum)](http://nlp.h-its.org/bpemb/tet) ・ 
[tg (Tajik)](http://nlp.h-its.org/bpemb/tg) ・ 
[th (Thai)](http://nlp.h-its.org/bpemb/th) ・ 
[ti (Tigrinya)](http://nlp.h-its.org/bpemb/ti) ・ 
[tk (Turkmen)](http://nlp.h-its.org/bpemb/tk) ・ 
[tl (Tagalog)](http://nlp.h-its.org/bpemb/tl) ・ 
[tn (Tswana)](http://nlp.h-its.org/bpemb/tn) ・ 
[to (Tonga)](http://nlp.h-its.org/bpemb/to) ・ 
[tpi (Tok Pisin)](http://nlp.h-its.org/bpemb/tpi) ・ 
[tr (Turkish)](http://nlp.h-its.org/bpemb/tr) ・ 
[ts (Tsonga)](http://nlp.h-its.org/bpemb/ts) ・ 
[tt (Tatar)](http://nlp.h-its.org/bpemb/tt) ・ 
[tum (Tumbuka)](http://nlp.h-its.org/bpemb/tum) ・ 
[tw (Twi)](http://nlp.h-its.org/bpemb/tw) ・ 
[ty (Tahitian)](http://nlp.h-its.org/bpemb/ty) ・ 
[tyv (Tuvinian)](http://nlp.h-its.org/bpemb/tyv)

[udm (Udmurt)](http://nlp.h-its.org/bpemb/udm) ・ 
[ug (Uighur)](http://nlp.h-its.org/bpemb/ug) ・ 
[uk (Ukrainian)](http://nlp.h-its.org/bpemb/uk) ・ 
[ur (Urdu)](http://nlp.h-its.org/bpemb/ur) ・ 
[uz (Uzbek)](http://nlp.h-its.org/bpemb/uz)

[ve (Venda)](http://nlp.h-its.org/bpemb/ve) ・ 
[vec (Venetian)](http://nlp.h-its.org/bpemb/vec) ・ 
[vep (Veps)](http://nlp.h-its.org/bpemb/vep) ・ 
[vi (Vietnamese)](http://nlp.h-its.org/bpemb/vi) ・ 
[vls (Vlaams)](http://nlp.h-its.org/bpemb/vls) ・ 
[vo (Volapük)](http://nlp.h-its.org/bpemb/vo)

[wa (Walloon)](http://nlp.h-its.org/bpemb/wa) ・ 
[war (Waray)](http://nlp.h-its.org/bpemb/war) ・ 
[wo (Wolof)](http://nlp.h-its.org/bpemb/wo) ・ 
[wuu (Wu Chinese)](http://nlp.h-its.org/bpemb/wuu)

[xal (Kalmyk)](http://nlp.h-its.org/bpemb/xal) ・ 
[xh (Xhosa)](http://nlp.h-its.org/bpemb/xh) ・ 
[xmf (Mingrelian)](http://nlp.h-its.org/bpemb/xmf)

[yi (Yiddish)](http://nlp.h-its.org/bpemb/yi) ・ 
[yo (Yoruba)](http://nlp.h-its.org/bpemb/yo)

[za (Zhuang)](http://nlp.h-its.org/bpemb/za) ・ 
[zea (Zeeuws)](http://nlp.h-its.org/bpemb/zea) ・ 
[zh (Chinese)](http://nlp.h-its.org/bpemb/zh) ・ 
[zu (Zulu)](http://nlp.h-its.org/bpemb/zu)


## Citing BPEmb

If you use BPEmb in academic work, please cite:

```
@InProceedings{heinzerling2018bpemb,
  author = {Benjamin Heinzerling and Michael Strube},
  title = "{BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages}",
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {May 7-12, 2018},
  address = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {979-10-95546-00-9},
  language = {english}
  }
```
