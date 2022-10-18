# Harvesting Linguistically Related Languages for Low-Resource NMT

## Overview
Large Language Models have revolutionized the field of Natural Language Processing, with Transformers quickly becoming the prevalent choice for practitioners. Consequently, great advances have been made in Neural Machine Translation, with a notable increase in terms of BLEU score compared to the pre-Transformer era. Despite this success, the low-resource setting continues to pose significant challenges. Back-translation and knowledge distillation attempt to solve this issue, yet the model selection, to either generate the data or distill the knowledge, still remains an open question. In this project, we investigate whether using multiple bi-lingual models can capture different but related linguistic information and, as a result, improve the models' performance.

## Setup
We provide a conda environment for the installation of the required packages.
```bash
conda env create -f env.yml
```

## Project Structure
```
├── datamodules # PyTorch Lightning DataModules
│   ├── ccmatrix.py # CCMatrix Dataset DataModule
│   └── mt_distillation_datamodule.py # MT Distillation Dataset DataModule
├── distiller # Distillation Module
│   ├── distiller_lightning_module.py # Distillation Lightning Module
│   └── train_distiller.py # Distillation Training Script
├── env.yml # Conda Environment
├── experiments # Experiment Folder
│   ├── barplot.ipynb # Barplot Notebook
│   ├── eval_results # Evaluation Results
│   ├── evaluate_models.py # Evaluation Script
│   └── table.ipynb # Table Notebook
├── images # Images
├── lisa # lisa cluster job scripts
├── lisa_cheatsheet.md # lisa cluster cheatsheet
├── README.md # README
└── scripts # Scripts
    ├── data_analysis.py # Data Analysis Script
    ├── finetune.py # Finetuning Script
    ├── generate_synthetic_data.py # Synthetic Data Generation Script
    ├── init_datasets.py # Dataset Initialization Script
    ├── lstm # LSTM Baseline
    │   ├── build_vocab.py # Vocabulary Building Script
    │   ├── config.yaml # Configuration File
    │   └── download_dataset.py # Dataset Download Script
    ├── perplexity.py # Perplexity Calculation Script
    ├── sample_ccmatrix.py # CCMatrix Sampling Script
    └── synthesis_merge.py # Synthetic Data Merging Script
```

## Results
The results can be fully replicated by running the notebooks [experiments/barplot.ipynb](experiments/barplot.ipynb), [experiments/evaluate_models.py](experiments/evaluate_models.py) and [experiments/table.ipynb](experiments/table.ipynb) in the [experiments](experiments) folder.
<img src="https://github.com/j0hngou/LRNMT/blob/master/images/results.png" width="30%" height="50%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/j0hngou/LRNMT/blob/master/images/barplot.png" width="50%" height="50%">

## Training
The distilled model can be obtain by running the following command:
```
python distiller/train_distiller.py
```

A fine-tuned model on Italian can be obtained by running the following command:
```
python scripts/finetune.py --language_name "Italian" --code "it"
```
A synthetic dataset from T5-base can be obtained by running the following command:
```
python scripts/generate_synthetic_data.py --model_name "t5-base"
```

#
The project was developed for the MSc AI course "Deep Learning for Natural Language Processing" at the University of Amsterdam.
