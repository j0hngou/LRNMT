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
├── datamodules
│   ├── ccmatrix.py
│   └── mt_distillation_datamodule.py
├── distiller
│   ├── distiller_lightning_module.py
│   └── train_distiller.py
├── env.yml
├── experiments
│   ├── barplot.ipynb
│   ├── eval_results
│   ├── evaluate_models.py
│   └── table.ipynb
├── images
├── lisa
├── lisa_cheatsheet.md
├── README.md
└── scripts
    ├── data_analysis.py
    ├── finetune.py
    ├── generate_synthetic_data.py
    ├── init_datasets.py
    ├── lstm
    │   ├── build_vocab.py
    │   ├── config.yaml
    │   ├── download_dataset.py
    │   └── validations.txt
    ├── perplexity.py
    ├── sample_ccmatrix.py
    └── synthesis_merge.py 
```

## Results
The results can be fully replicated by running the notebooks in the experiments folder.

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

