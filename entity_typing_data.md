# Entity typing data in 275 languages

For our entity typing experiments, we collected entity *labels* (Wikipedia article titles) from [wikidata](https://www.wikidata.org) and mapped them to fine-grained *entity types*. The resulting data in 275 languages can be downloaded [here](http://cosyne.h-its.org/bpemb/wikidata_to_type_map).

We provide data for several entity type inventories:

## Notable FIGER type (filename: label_figer_notable_type)

A map from wikidata labels to exactly one of the 112 fine-grained entity types introduced by [Ling & Weld, 2012](http://aiweb.cs.washington.edu/ai/pubs/ling-aaai12.pdf). The entity type is selected via the *notable type* property in Freebase. This is the data on which the results reported in our [paper](https://arxiv.org/pdf/1710.02187.pdf) are based.

## FIGER types (filename: label_figer_types)

A map from wikidata labels to one or more of the 112 FIGER types. The entity types are all types found for the given entity in Freebase.

## Gillick types (filename: label_gillick_types)

A map from wikidata labels to one or more of the 89 entity types proposed by [Gillick et al., 2014](https://arxiv.org/abs/1412.1820). The entity types are all types found for the given entity in Freebase.


## Gillick 4class types (filename: label_gillick_4class_types)


A map from wikidata labels to one or more of the four basic entity types PERSON, LOCATION, ORGANIZATION, OTHER, via the mapping by [Gillick et al., 2014](https://arxiv.org/abs/1412.1820). The entity types are mapped from all types found for the given entity in Freebase.
