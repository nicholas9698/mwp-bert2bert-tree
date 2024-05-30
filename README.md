# BERT2BERT-Tree
This is the official repo for the paper "[A Fusion Network of Sequence-to-Sequence Model and Tree-based Decoder for Math Word Problems](https://arxiv.org/)".

## Train and Test

> The checkpoints of MWP-BERT we used can be obtained from https://drive.google.com/drive/folders/1staS6zCiAarNNMz9GhckNGkc_OD2HL1O?usp=drive_link
> 
> *Liang, Zhengwen et al.*, 
> *[MWP-BERT: Numeracy-Augmented Pre-training for Math Word Problem Solving](https://aclanthology.org/2022.findings-naacl.74)*,
> *NAACL 2022*

### Math23K

Run the following to train the model and reproduce the results of **Train-Test**:
```
python -u run_bert2bert_tree_attributes.py
```

Run the following to train the model and reproduce the results of **5-Fold Cross-Validation**:

```
python -u cross_valid_bert2bert_tree.py
```

### Ape-clean

Run the following to train the model and reproduce the results of **Train-Test**:
```
python -u run_wape.py
```

### Main Results

We reproduce the main results of **BERT2BERT-Tree** in the following tables:

| Dataset | Euqation accuracy | Value accuracy |
| :--- | :---: | :---: |
| Math23K (train/test) | 73.2 | 85.9 |
| Math23K (5-fold cross-validation) | 72.7 | 84.7 |
| Math23K (train with Ape-clean) | 77.7 | 91.5 |
| Ape-clean (train/test) | 68.0 | 81.8 |

## Dataset

The data used by the bert2bert model is uploaded to `data/math23k/`.
> *Wang, Yan, et al.*,
> *[Deep Neural Solver for Math Word Problems](https://doi.org/10.18653/v1/D17-1088)* *(Math23K first introduced)*,
> *EMNLP 2017*

The [Math23K](https://github.com/2003pro/Graph2Tree/tree/master/math23k/data) and [Ape-clean](https://github.com/LZhenwen/MWP-BERT/tree/main/Fine-tuning/Math23k/data) data for bert2bert-tree is in `data/`. 

Moreover, the elements of [Ape-clean](https://github.com/LZhenwen/MWP-BERT/tree/main/Fine-tuning/Math23k/data) are sampled from [Ape-210K](https://github.com/Chenny0808/ape210k) as listed in `(data/ape_simple_id.txt)` and `(data/ape_simple_test_id.txt)`
> *Zhao, Wei, et al.*,
> *[Ape210K: A Large-Scale and Template-Rich Dataset of Math Word Problems](https://doi.org/10.48550/arXiv.2009.11506)*,
> *arXiv:2009.11506*

## Entity Attributes Building

Build the entity attributes for your data:

> The process relies on *[Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)*, follow this *[guide](https://github.com/Lynten/stanford-corenlp)* to install the PyPI package and download the Chinese model package.

1. Follow the instruction ***Recut the word*** in `attributes_building.ipynb` to prepare the data of entity attributes generating.

2. Execute the ***Extract entity attributes of numbers*** in `attributes_building.ipynb` to generate the corresponding number entity attributes.

## Citation

If you find this work useful, please cite our paper:
```
@article{}
```
