# FPSR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.76097887.svg)](https://doi.org/10.5281/zenodo.7609787)
[![RecBole](https://img.shields.io/badge/RecBole-1.1.1-orange)](https://recbole.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2207.05959-red)](https://arxiv.org/abs/2207.05959) 
[![License](https://img.shields.io/github/license/Joinn99/FPSR)](https://github.com/Joinn99/FPSR/blob/master/LICENSE.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joinn99/FPSR/blob/torch/FPSR%20Demonstration.ipynb)

**PyTorch version** (Default) | [**CuPy version**](https://github.com/Joinn99/FPSR/)

This is the official implementation of our *ACM The Web Conference 2023 (WWW 2023)* paper:
> Tianjun Wei, Jianghong Ma, Tommy W.S. Chow. Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation. [[arXiv](https://arxiv.org/abs/2207.05959)]

<p align="center" width="100%">
    <kbd><img width="100%" src="https://raw.githubusercontent.com/Joinn99/RepositoryResource/master/FPSR.svg"> </kbd>
</p>

## Requirements
The model implementation ensures compatibility with the Recommendation Toolbox [RecBole](https://recbole.io/) (Github: [Recbole](https://github.com/RUCAIBox/RecBole)). This is the pure PyTorch version of FPSR model. 

The requirements of the running environement:

- Python: 3.8+
- RecBole: 1.1.1
- PyTorch: 1.9.0+

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
The script `run.py` is used to reproduced the results presented in paper. Train and avaluate FPSR on a specific dataset, run
```bash
python run.py --dataset DATASET_NAME
```
## Google Colab
We also provide Colab notebook version of FPSR, you can click [here](https://colab.research.google.com/github/Joinn99/FPSR/blob/torch/FPSR%20Demonstration.ipynb) to open Google Colab, select the runtime type as *GPU*, and run the model.

## Citation
If you wish, please cite the following paper:

```bibtex
@misc{FPSR,
  doi = {10.48550/ARXIV.2207.05959},
  author = {Wei, Tianjun and Ma, Jianghong and Chow, Tommy W. S.},
  title = {Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation},
  year = {2023},
}
```