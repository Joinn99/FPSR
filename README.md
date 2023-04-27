# FPSR
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3543507.3583240-ff69b4)](https://doi.org/10.1145/3543507.3583240)
[![RecBole](https://img.shields.io/badge/RecBole-1.1.1-orange)](https://recbole.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2207.05959-red)](https://arxiv.org/abs/2207.05959) 
[![License](https://img.shields.io/github/license/Joinn99/FPSR)](https://github.com/Joinn99/FPSR/blob/cupy/LICENSE.md)

[**PyTorch version**](https://github.com/Joinn99/FPSR/tree/torch) (Default) | **CuPy version**

This is the official implementation of our *ACM The Web Conference 2023 (WWW 2023)* paper:
> Tianjun Wei, Jianghong Ma, Tommy W.S. Chow. Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation. [[arXiv](https://arxiv.org/abs/2207.05959)]

<p align="center" width="100%">
    <kbd><img width="100%" src="https://raw.githubusercontent.com/Joinn99/RepositoryResource/master/FPSR.svg"> </kbd>
</p>

## Requirements
The model implementation ensures compatibility with the Recommendation Toolbox [RecBole](https://recbole.io/) (Github: [Recbole](https://github.com/RUCAIBox/RecBole)). This version employs [CuPy](https://cupy.dev/) for matrix storage and computation, and produces the results in the paper. In addition, we also offer a pure PyTorch version of FPSR model, which can be found [here](https://github.com/Joinn99/FPSR/tree/torch). 

The requirements of the running environement:

- Python: 3.8+
- RecBole: 1.1.1
- CuPy: 0.10.5+

## Dataset
Here we only put zip files of datasets in the respository due to the storage limits. To use the dataset, run
```bash
unzip -o "Data/*.zip"
```
If you like to test FPSR on the custom dataset, please place the dataset files in the following path:
```bash
.
|-Data
| |-[CUSTOM_DATASET_NAME]
| | |-[CUSTOM_DATASET_NAME].user
| | |-[CUSTOM_DATASET_NAME].item
| | |-[CUSTOM_DATASET_NAME].inter

```
And create `[CUSTOM_DATASET_NAME].yaml` in `./Params` with the following content:
```yaml
dataset: [CUSTOM_DATASET_NAME]
```

For the format of each dataset file, please refer to [RecBole API](https://recbole.io/docs/user_guide/data/atomic_files.html).

## Hyperparameter
For each dataset, the optimal hyperparameters are stored in `Params/[DATASET].yaml`. To tune the hyperparamters, modify the corresponding values in the file for each dataset.

## Running
The script `run.py` is used to reproduce the results presented in paper. Train and evaluate FPSR on a specific dataset, run
```bash
python run.py --dataset DATASET_NAME
```

## Citation
If you wish, please cite the following paper:

```bibtex
@inproceedings{10.1145/3543507.3583240,
author = {Wei, Tianjun and Ma, Jianghong and Chow, Tommy W. S.},
title = {Fine-Tuning Partition-Aware Item Similarities for Efficient and Scalable Recommendation},
year = {2023},
isbn = {9781450394161},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3543507.3583240},
doi = {10.1145/3543507.3583240},
pages = {823–832},
numpages = {10},
keywords = {Recommender System, Similarity Measuring, Collaborative Filtering, Graph Partitioning},
location = {Austin, TX, USA},
series = {WWW '23}
}
```