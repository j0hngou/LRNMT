from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from pathlib import Path

dataset = load_dataset("j0hngou/ccmatrix_en-it")

low_resource_dev = load_dataset(path="j0hngou/ccmatrix_en-it", split="train[:1500]")
low_resource_test = load_dataset(path="j0hngou/ccmatrix_en-it", split="train[1500:3000]")
low_resource_train = load_dataset(path="j0hngou/ccmatrix_en-it", split="train[3000:15000]")

dataset_dict = DatasetDict({ 
  "train": low_resource_train,
  "validation": low_resource_dev,
  "test": low_resource_test
})


data_dir = "./ccmatrix_en-it"
dataset_dict.save_to_disk(data_dir)