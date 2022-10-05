# FPSR
This is the official implementation of our paper:
> Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation

## Requirements
The model implementation ensures compatibility with the Recommendation Toolbox [RecBole](https://recbole.io/) (Github: [Recbole](https://github.com/RUCAIBox/RecBole)), and uses [CuPy](https://cupy.dev/) to accrelate the model training on GPU. The requirements of the running environement:
- Python: 3.8+
- RecBole: 1.0.1
- CuPy: 0.10.5+

## Dataset
Here we only put zip files of datasets in the respository due to the storage limits. To use the dataset, run
```bash
unzip -o "Data/*.zip"
```

## Run
The script `run.py` is used to reproduced the results presented in paper. Train and avaluate FPSR on a specific dataset, run
```bash
python run.py --dataset DATASET_NAME
```
To produce all results, change the `seed` parameter in `Params/Overall.yaml` to any element in [2020, 2022, 2024, 2026, 2028, 2030, 2032] to produce all results.

