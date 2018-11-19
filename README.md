# BPEmb

BPEmb is a collection of pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. Its intended use is as input for neural models in natural language processing.

[Website](http://cosyne.h-its.org/bpemb)

## Usage

Install BPEmb with pip:

```bash
pip install bpemb
```

Embeddings and SentencePiece models will be downloaded automatically the first time you use them.

```python
>>> from bpemb install BPEmb
# load English BPEmb model with vocabulary size 50k and 300-dimensional embeddings
>>> bpemb_en = BPEmb(lang="en", vs=50000, dim=300)
downloading http://cosyne.h-its.org/bpemb/en/en.wiki.bpe.vs50000.model
downloading http://cosyne.h-its.org/bpemb/en/en.wiki.bpe.vs50000.d300.w2v.bin.tar.gz
```

The two main things you can do with BPEmb are subword segmentation:
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

...and using pretrained subword embeddings:
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
(50000, 300)
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

[aa (Afar)](http://cosyne.h-its.org/bpemb/aa) ・ 
[ab (Abkhazian)](http://cosyne.h-its.org/bpemb/ab) ・ 
[ace (Achinese)](http://cosyne.h-its.org/bpemb/ace) ・ 
[ady (Adyghe)](http://cosyne.h-its.org/bpemb/ady) ・ 
[af (Afrikaans)](http://cosyne.h-its.org/bpemb/af) ・ 
[ak (Akan)](http://cosyne.h-its.org/bpemb/ak) ・ 
[als (Tosk Albanian)](http://cosyne.h-its.org/bpemb/als) ・ 
[am (Amharic)](http://cosyne.h-its.org/bpemb/am) ・ 
[an (Aragonese)](http://cosyne.h-its.org/bpemb/an) ・ 
[ang (Old English)](http://cosyne.h-its.org/bpemb/ang) ・ 
[ar (Arabic)](http://cosyne.h-its.org/bpemb/ar) ・ 
[arc (Official Aramaic)](http://cosyne.h-its.org/bpemb/arc) ・ 
[arz (Egyptian Arabic)](http://cosyne.h-its.org/bpemb/arz) ・ 
[as (Assamese)](http://cosyne.h-its.org/bpemb/as) ・ 
[ast (Asturian)](http://cosyne.h-its.org/bpemb/ast) ・ 
[atj (Atikamekw)](http://cosyne.h-its.org/bpemb/atj) ・ 
[av (Avaric)](http://cosyne.h-its.org/bpemb/av) ・ 
[ay (Aymara)](http://cosyne.h-its.org/bpemb/ay) ・ 
[az (Azerbaijani)](http://cosyne.h-its.org/bpemb/az) ・ 
[azb (South Azerbaijani)](http://cosyne.h-its.org/bpemb/azb)

[ba (Bashkir)](http://cosyne.h-its.org/bpemb/ba) ・ 
[bar (Bavarian)](http://cosyne.h-its.org/bpemb/bar) ・ 
[bcl (Central Bikol)](http://cosyne.h-its.org/bpemb/bcl) ・ 
[be (Belarusian)](http://cosyne.h-its.org/bpemb/be) ・ 
[bg (Bulgarian)](http://cosyne.h-its.org/bpemb/bg) ・ 
[bh (Bihari languages)](http://cosyne.h-its.org/bpemb/bh) ・ 
[bi (Bislama)](http://cosyne.h-its.org/bpemb/bi) ・ 
[bjn (Banjar)](http://cosyne.h-its.org/bpemb/bjn) ・ 
[bm (Bambara)](http://cosyne.h-its.org/bpemb/bm) ・ 
[bn (Bengali)](http://cosyne.h-its.org/bpemb/bn) ・ 
[bo (Tibetan)](http://cosyne.h-its.org/bpemb/bo) ・ 
[bpy (Bishnupriya)](http://cosyne.h-its.org/bpemb/bpy) ・ 
[br (Breton)](http://cosyne.h-its.org/bpemb/br) ・ 
[bs (Bosnian)](http://cosyne.h-its.org/bpemb/bs) ・ 
[bug (Buginese)](http://cosyne.h-its.org/bpemb/bug) ・ 
[bxr (Russia Buriat)](http://cosyne.h-its.org/bpemb/bxr)

[ca (Catalan)](http://cosyne.h-its.org/bpemb/ca) ・ 
[cdo (Min Dong Chinese)](http://cosyne.h-its.org/bpemb/cdo) ・ 
[ce (Chechen)](http://cosyne.h-its.org/bpemb/ce) ・ 
[ceb (Cebuano)](http://cosyne.h-its.org/bpemb/ceb) ・ 
[ch (Chamorro)](http://cosyne.h-its.org/bpemb/ch) ・ 
[cho (Choctaw)](http://cosyne.h-its.org/bpemb/cho) ・ 
[chr (Cherokee)](http://cosyne.h-its.org/bpemb/chr) ・ 
[chy (Cheyenne)](http://cosyne.h-its.org/bpemb/chy) ・ 
[ckb (Central Kurdish)](http://cosyne.h-its.org/bpemb/ckb) ・ 
[co (Corsican)](http://cosyne.h-its.org/bpemb/co) ・ 
[cr (Cree)](http://cosyne.h-its.org/bpemb/cr) ・ 
[crh (Crimean Tatar)](http://cosyne.h-its.org/bpemb/crh) ・ 
[cs (Czech)](http://cosyne.h-its.org/bpemb/cs) ・ 
[csb (Kashubian)](http://cosyne.h-its.org/bpemb/csb) ・ 
[cu (Church Slavic)](http://cosyne.h-its.org/bpemb/cu) ・ 
[cv (Chuvash)](http://cosyne.h-its.org/bpemb/cv) ・ 
[cy (Welsh)](http://cosyne.h-its.org/bpemb/cy)

[da (Danish)](http://cosyne.h-its.org/bpemb/da) ・ 
[de (German)](http://cosyne.h-its.org/bpemb/de) ・ 
[din (Dinka)](http://cosyne.h-its.org/bpemb/din) ・ 
[diq (Dimli)](http://cosyne.h-its.org/bpemb/diq) ・ 
[dsb (Lower Sorbian)](http://cosyne.h-its.org/bpemb/dsb) ・ 
[dty (Dotyali)](http://cosyne.h-its.org/bpemb/dty) ・ 
[dv (Dhivehi)](http://cosyne.h-its.org/bpemb/dv) ・ 
[dz (Dzongkha)](http://cosyne.h-its.org/bpemb/dz)

[ee (Ewe)](http://cosyne.h-its.org/bpemb/ee) ・ 
[el (Modern Greek)](http://cosyne.h-its.org/bpemb/el) ・ 
[en (English)](http://cosyne.h-its.org/bpemb/en) ・ 
[eo (Esperanto)](http://cosyne.h-its.org/bpemb/eo) ・ 
[es (Spanish)](http://cosyne.h-its.org/bpemb/es) ・ 
[et (Estonian)](http://cosyne.h-its.org/bpemb/et) ・ 
[eu (Basque)](http://cosyne.h-its.org/bpemb/eu) ・ 
[ext (Extremaduran)](http://cosyne.h-its.org/bpemb/ext)

[fa (Persian)](http://cosyne.h-its.org/bpemb/fa) ・ 
[ff (Fulah)](http://cosyne.h-its.org/bpemb/ff) ・ 
[fi (Finnish)](http://cosyne.h-its.org/bpemb/fi) ・ 
[fj (Fijian)](http://cosyne.h-its.org/bpemb/fj) ・ 
[fo (Faroese)](http://cosyne.h-its.org/bpemb/fo) ・ 
[fr (French)](http://cosyne.h-its.org/bpemb/fr) ・ 
[frp (Arpitan)](http://cosyne.h-its.org/bpemb/frp) ・ 
[frr (Northern Frisian)](http://cosyne.h-its.org/bpemb/frr) ・ 
[fur (Friulian)](http://cosyne.h-its.org/bpemb/fur) ・ 
[fy (Western Frisian)](http://cosyne.h-its.org/bpemb/fy)

[ga (Irish)](http://cosyne.h-its.org/bpemb/ga) ・ 
[gag (Gagauz)](http://cosyne.h-its.org/bpemb/gag) ・ 
[gan (Gan Chinese)](http://cosyne.h-its.org/bpemb/gan) ・ 
[gd (Scottish Gaelic)](http://cosyne.h-its.org/bpemb/gd) ・ 
[gl (Galician)](http://cosyne.h-its.org/bpemb/gl) ・ 
[glk (Gilaki)](http://cosyne.h-its.org/bpemb/glk) ・ 
[gn (Guarani)](http://cosyne.h-its.org/bpemb/gn) ・ 
[gom (Goan Konkani)](http://cosyne.h-its.org/bpemb/gom) ・ 
[got (Gothic)](http://cosyne.h-its.org/bpemb/got) ・ 
[gu (Gujarati)](http://cosyne.h-its.org/bpemb/gu) ・ 
[gv (Manx)](http://cosyne.h-its.org/bpemb/gv)

[ha (Hausa)](http://cosyne.h-its.org/bpemb/ha) ・ 
[hak (Hakka Chinese)](http://cosyne.h-its.org/bpemb/hak) ・ 
[haw (Hawaiian)](http://cosyne.h-its.org/bpemb/haw) ・ 
[he (Hebrew)](http://cosyne.h-its.org/bpemb/he) ・ 
[hi (Hindi)](http://cosyne.h-its.org/bpemb/hi) ・ 
[hif (Fiji Hindi)](http://cosyne.h-its.org/bpemb/hif) ・ 
[ho (Hiri Motu)](http://cosyne.h-its.org/bpemb/ho) ・ 
[hr (Croatian)](http://cosyne.h-its.org/bpemb/hr) ・ 
[hsb (Upper Sorbian)](http://cosyne.h-its.org/bpemb/hsb) ・ 
[ht (Haitian)](http://cosyne.h-its.org/bpemb/ht) ・ 
[hu (Hungarian)](http://cosyne.h-its.org/bpemb/hu) ・ 
[hy (Armenian)](http://cosyne.h-its.org/bpemb/hy) ・ 
[hz (Herero)](http://cosyne.h-its.org/bpemb/hz)

[ia (Interlingua)](http://cosyne.h-its.org/bpemb/ia) ・ 
[id (Indonesian)](http://cosyne.h-its.org/bpemb/id) ・ 
[ie (Interlingue)](http://cosyne.h-its.org/bpemb/ie) ・ 
[ig (Igbo)](http://cosyne.h-its.org/bpemb/ig) ・ 
[ii (Sichuan Yi)](http://cosyne.h-its.org/bpemb/ii) ・ 
[ik (Inupiaq)](http://cosyne.h-its.org/bpemb/ik) ・ 
[ilo (Iloko)](http://cosyne.h-its.org/bpemb/ilo) ・ 
[io (Ido)](http://cosyne.h-its.org/bpemb/io) ・ 
[is (Icelandic)](http://cosyne.h-its.org/bpemb/is) ・ 
[it (Italian)](http://cosyne.h-its.org/bpemb/it) ・ 
[iu (Inuktitut)](http://cosyne.h-its.org/bpemb/iu)

[ja (Japanese)](http://cosyne.h-its.org/bpemb/ja) ・ 
[jam (Jamaican Creole English)](http://cosyne.h-its.org/bpemb/jam) ・ 
[jbo (Lojban)](http://cosyne.h-its.org/bpemb/jbo) ・ 
[jv (Javanese)](http://cosyne.h-its.org/bpemb/jv)

[ka (Georgian)](http://cosyne.h-its.org/bpemb/ka) ・ 
[kaa (Kara-Kalpak)](http://cosyne.h-its.org/bpemb/kaa) ・ 
[kab (Kabyle)](http://cosyne.h-its.org/bpemb/kab) ・ 
[kbd (Kabardian)](http://cosyne.h-its.org/bpemb/kbd) ・ 
[kbp (Kabiyè)](http://cosyne.h-its.org/bpemb/kbp) ・ 
[kg (Kongo)](http://cosyne.h-its.org/bpemb/kg) ・ 
[ki (Kikuyu)](http://cosyne.h-its.org/bpemb/ki) ・ 
[kj (Kuanyama)](http://cosyne.h-its.org/bpemb/kj) ・ 
[kk (Kazakh)](http://cosyne.h-its.org/bpemb/kk) ・ 
[kl (Kalaallisut)](http://cosyne.h-its.org/bpemb/kl) ・ 
[km (Central Khmer)](http://cosyne.h-its.org/bpemb/km) ・ 
[kn (Kannada)](http://cosyne.h-its.org/bpemb/kn) ・ 
[ko (Korean)](http://cosyne.h-its.org/bpemb/ko) ・ 
[koi (Komi-Permyak)](http://cosyne.h-its.org/bpemb/koi) ・ 
[kr (Kanuri)](http://cosyne.h-its.org/bpemb/kr) ・ 
[krc (Karachay-Balkar)](http://cosyne.h-its.org/bpemb/krc) ・ 
[ks (Kashmiri)](http://cosyne.h-its.org/bpemb/ks) ・ 
[ksh (Kölsch)](http://cosyne.h-its.org/bpemb/ksh) ・ 
[ku (Kurdish)](http://cosyne.h-its.org/bpemb/ku) ・ 
[kv (Komi)](http://cosyne.h-its.org/bpemb/kv) ・ 
[kw (Cornish)](http://cosyne.h-its.org/bpemb/kw) ・ 
[ky (Kirghiz)](http://cosyne.h-its.org/bpemb/ky)

[la (Latin)](http://cosyne.h-its.org/bpemb/la) ・ 
[lad (Ladino)](http://cosyne.h-its.org/bpemb/lad) ・ 
[lb (Luxembourgish)](http://cosyne.h-its.org/bpemb/lb) ・ 
[lbe (Lak)](http://cosyne.h-its.org/bpemb/lbe) ・ 
[lez (Lezghian)](http://cosyne.h-its.org/bpemb/lez) ・ 
[lg (Ganda)](http://cosyne.h-its.org/bpemb/lg) ・ 
[li (Limburgan)](http://cosyne.h-its.org/bpemb/li) ・ 
[lij (Ligurian)](http://cosyne.h-its.org/bpemb/lij) ・ 
[lmo (Lombard)](http://cosyne.h-its.org/bpemb/lmo) ・ 
[ln (Lingala)](http://cosyne.h-its.org/bpemb/ln) ・ 
[lo (Lao)](http://cosyne.h-its.org/bpemb/lo) ・ 
[lrc (Northern Luri)](http://cosyne.h-its.org/bpemb/lrc) ・ 
[lt (Lithuanian)](http://cosyne.h-its.org/bpemb/lt) ・ 
[ltg (Latgalian)](http://cosyne.h-its.org/bpemb/ltg) ・ 
[lv (Latvian)](http://cosyne.h-its.org/bpemb/lv)

[mai (Maithili)](http://cosyne.h-its.org/bpemb/mai) ・ 
[mdf (Moksha)](http://cosyne.h-its.org/bpemb/mdf) ・ 
[mg (Malagasy)](http://cosyne.h-its.org/bpemb/mg) ・ 
[mh (Marshallese)](http://cosyne.h-its.org/bpemb/mh) ・ 
[mhr (Eastern Mari)](http://cosyne.h-its.org/bpemb/mhr) ・ 
[mi (Maori)](http://cosyne.h-its.org/bpemb/mi) ・ 
[min (Minangkabau)](http://cosyne.h-its.org/bpemb/min) ・ 
[mk (Macedonian)](http://cosyne.h-its.org/bpemb/mk) ・ 
[ml (Malayalam)](http://cosyne.h-its.org/bpemb/ml) ・ 
[mn (Mongolian)](http://cosyne.h-its.org/bpemb/mn) ・ 
[mr (Marathi)](http://cosyne.h-its.org/bpemb/mr) ・ 
[mrj (Western Mari)](http://cosyne.h-its.org/bpemb/mrj) ・ 
[ms (Malay)](http://cosyne.h-its.org/bpemb/ms) ・ 
[mt (Maltese)](http://cosyne.h-its.org/bpemb/mt) ・ 
[mus (Creek)](http://cosyne.h-its.org/bpemb/mus) ・ 
[mwl (Mirandese)](http://cosyne.h-its.org/bpemb/mwl) ・ 
[my (Burmese)](http://cosyne.h-its.org/bpemb/my) ・ 
[myv (Erzya)](http://cosyne.h-its.org/bpemb/myv) ・ 
[mzn (Mazanderani)](http://cosyne.h-its.org/bpemb/mzn)

[na (Nauru)](http://cosyne.h-its.org/bpemb/na) ・ 
[nah (Nahuatl languages)](http://cosyne.h-its.org/bpemb/nah) ・ 
[nap (Neapolitan)](http://cosyne.h-its.org/bpemb/nap) ・ 
[nds (Low German)](http://cosyne.h-its.org/bpemb/nds) ・ 
[ne (Nepali)](http://cosyne.h-its.org/bpemb/ne) ・ 
[new (Newari)](http://cosyne.h-its.org/bpemb/new) ・ 
[ng (Ndonga)](http://cosyne.h-its.org/bpemb/ng) ・ 
[nl (Dutch)](http://cosyne.h-its.org/bpemb/nl) ・ 
[nn (Norwegian Nynorsk)](http://cosyne.h-its.org/bpemb/nn) ・ 
[no (Norwegian)](http://cosyne.h-its.org/bpemb/no) ・ 
[nov (Novial)](http://cosyne.h-its.org/bpemb/nov) ・ 
[nrm (Narom)](http://cosyne.h-its.org/bpemb/nrm) ・ 
[nso (Pedi)](http://cosyne.h-its.org/bpemb/nso) ・ 
[nv (Navajo)](http://cosyne.h-its.org/bpemb/nv) ・ 
[ny (Nyanja)](http://cosyne.h-its.org/bpemb/ny)

[oc (Occitan)](http://cosyne.h-its.org/bpemb/oc) ・ 
[olo (Livvi)](http://cosyne.h-its.org/bpemb/olo) ・ 
[om (Oromo)](http://cosyne.h-its.org/bpemb/om) ・ 
[or (Oriya)](http://cosyne.h-its.org/bpemb/or) ・ 
[os (Ossetian)](http://cosyne.h-its.org/bpemb/os)

[pa (Panjabi)](http://cosyne.h-its.org/bpemb/pa) ・ 
[pag (Pangasinan)](http://cosyne.h-its.org/bpemb/pag) ・ 
[pam (Pampanga)](http://cosyne.h-its.org/bpemb/pam) ・ 
[pap (Papiamento)](http://cosyne.h-its.org/bpemb/pap) ・ 
[pcd (Picard)](http://cosyne.h-its.org/bpemb/pcd) ・ 
[pdc (Pennsylvania German)](http://cosyne.h-its.org/bpemb/pdc) ・ 
[pfl (Pfaelzisch)](http://cosyne.h-its.org/bpemb/pfl) ・ 
[pi (Pali)](http://cosyne.h-its.org/bpemb/pi) ・ 
[pih (Pitcairn-Norfolk)](http://cosyne.h-its.org/bpemb/pih) ・ 
[pl (Polish)](http://cosyne.h-its.org/bpemb/pl) ・ 
[pms (Piemontese)](http://cosyne.h-its.org/bpemb/pms) ・ 
[pnb (Western Panjabi)](http://cosyne.h-its.org/bpemb/pnb) ・ 
[pnt (Pontic)](http://cosyne.h-its.org/bpemb/pnt) ・ 
[ps (Pushto)](http://cosyne.h-its.org/bpemb/ps) ・ 
[pt (Portuguese)](http://cosyne.h-its.org/bpemb/pt)

[qu (Quechua)](http://cosyne.h-its.org/bpemb/qu)

[rm (Romansh)](http://cosyne.h-its.org/bpemb/rm) ・ 
[rmy (Vlax Romani)](http://cosyne.h-its.org/bpemb/rmy) ・ 
[rn (Rundi)](http://cosyne.h-its.org/bpemb/rn) ・ 
[ro (Romanian)](http://cosyne.h-its.org/bpemb/ro) ・ 
[ru (Russian)](http://cosyne.h-its.org/bpemb/ru) ・ 
[rue (Rusyn)](http://cosyne.h-its.org/bpemb/rue) ・ 
[rw (Kinyarwanda)](http://cosyne.h-its.org/bpemb/rw)

[sa (Sanskrit)](http://cosyne.h-its.org/bpemb/sa) ・ 
[sah (Yakut)](http://cosyne.h-its.org/bpemb/sah) ・ 
[sc (Sardinian)](http://cosyne.h-its.org/bpemb/sc) ・ 
[scn (Sicilian)](http://cosyne.h-its.org/bpemb/scn) ・ 
[sco (Scots)](http://cosyne.h-its.org/bpemb/sco) ・ 
[sd (Sindhi)](http://cosyne.h-its.org/bpemb/sd) ・ 
[se (Northern Sami)](http://cosyne.h-its.org/bpemb/se) ・ 
[sg (Sango)](http://cosyne.h-its.org/bpemb/sg) ・ 
[sh (Serbo-Croatian)](http://cosyne.h-its.org/bpemb/sh) ・ 
[si (Sinhala)](http://cosyne.h-its.org/bpemb/si) ・ 
[sk (Slovak)](http://cosyne.h-its.org/bpemb/sk) ・ 
[sl (Slovenian)](http://cosyne.h-its.org/bpemb/sl) ・ 
[sm (Samoan)](http://cosyne.h-its.org/bpemb/sm) ・ 
[sn (Shona)](http://cosyne.h-its.org/bpemb/sn) ・ 
[so (Somali)](http://cosyne.h-its.org/bpemb/so) ・ 
[sq (Albanian)](http://cosyne.h-its.org/bpemb/sq) ・ 
[sr (Serbian)](http://cosyne.h-its.org/bpemb/sr) ・ 
[srn (Sranan Tongo)](http://cosyne.h-its.org/bpemb/srn) ・ 
[ss (Swati)](http://cosyne.h-its.org/bpemb/ss) ・ 
[st (Southern Sotho)](http://cosyne.h-its.org/bpemb/st) ・ 
[stq (Saterfriesisch)](http://cosyne.h-its.org/bpemb/stq) ・ 
[su (Sundanese)](http://cosyne.h-its.org/bpemb/su) ・ 
[sv (Swedish)](http://cosyne.h-its.org/bpemb/sv) ・ 
[sw (Swahili)](http://cosyne.h-its.org/bpemb/sw) ・ 
[szl (Silesian)](http://cosyne.h-its.org/bpemb/szl)

[ta (Tamil)](http://cosyne.h-its.org/bpemb/ta) ・ 
[tcy (Tulu)](http://cosyne.h-its.org/bpemb/tcy) ・ 
[te (Telugu)](http://cosyne.h-its.org/bpemb/te) ・ 
[tet (Tetum)](http://cosyne.h-its.org/bpemb/tet) ・ 
[tg (Tajik)](http://cosyne.h-its.org/bpemb/tg) ・ 
[th (Thai)](http://cosyne.h-its.org/bpemb/th) ・ 
[ti (Tigrinya)](http://cosyne.h-its.org/bpemb/ti) ・ 
[tk (Turkmen)](http://cosyne.h-its.org/bpemb/tk) ・ 
[tl (Tagalog)](http://cosyne.h-its.org/bpemb/tl) ・ 
[tn (Tswana)](http://cosyne.h-its.org/bpemb/tn) ・ 
[to (Tonga)](http://cosyne.h-its.org/bpemb/to) ・ 
[tpi (Tok Pisin)](http://cosyne.h-its.org/bpemb/tpi) ・ 
[tr (Turkish)](http://cosyne.h-its.org/bpemb/tr) ・ 
[ts (Tsonga)](http://cosyne.h-its.org/bpemb/ts) ・ 
[tt (Tatar)](http://cosyne.h-its.org/bpemb/tt) ・ 
[tum (Tumbuka)](http://cosyne.h-its.org/bpemb/tum) ・ 
[tw (Twi)](http://cosyne.h-its.org/bpemb/tw) ・ 
[ty (Tahitian)](http://cosyne.h-its.org/bpemb/ty) ・ 
[tyv (Tuvinian)](http://cosyne.h-its.org/bpemb/tyv)

[udm (Udmurt)](http://cosyne.h-its.org/bpemb/udm) ・ 
[ug (Uighur)](http://cosyne.h-its.org/bpemb/ug) ・ 
[uk (Ukrainian)](http://cosyne.h-its.org/bpemb/uk) ・ 
[ur (Urdu)](http://cosyne.h-its.org/bpemb/ur) ・ 
[uz (Uzbek)](http://cosyne.h-its.org/bpemb/uz)

[ve (Venda)](http://cosyne.h-its.org/bpemb/ve) ・ 
[vec (Venetian)](http://cosyne.h-its.org/bpemb/vec) ・ 
[vep (Veps)](http://cosyne.h-its.org/bpemb/vep) ・ 
[vi (Vietnamese)](http://cosyne.h-its.org/bpemb/vi) ・ 
[vls (Vlaams)](http://cosyne.h-its.org/bpemb/vls) ・ 
[vo (Volapük)](http://cosyne.h-its.org/bpemb/vo)

[wa (Walloon)](http://cosyne.h-its.org/bpemb/wa) ・ 
[war (Waray)](http://cosyne.h-its.org/bpemb/war) ・ 
[wo (Wolof)](http://cosyne.h-its.org/bpemb/wo) ・ 
[wuu (Wu Chinese)](http://cosyne.h-its.org/bpemb/wuu)

[xal (Kalmyk)](http://cosyne.h-its.org/bpemb/xal) ・ 
[xh (Xhosa)](http://cosyne.h-its.org/bpemb/xh) ・ 
[xmf (Mingrelian)](http://cosyne.h-its.org/bpemb/xmf)

[yi (Yiddish)](http://cosyne.h-its.org/bpemb/yi) ・ 
[yo (Yoruba)](http://cosyne.h-its.org/bpemb/yo)

[za (Zhuang)](http://cosyne.h-its.org/bpemb/za) ・ 
[zea (Zeeuws)](http://cosyne.h-its.org/bpemb/zea) ・ 
[zh (Chinese)](http://cosyne.h-its.org/bpemb/zh) ・ 
[zu (Zulu)](http://cosyne.h-its.org/bpemb/zu)



## Reference

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
