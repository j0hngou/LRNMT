from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional, List
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl
import torch


class MTDistillationDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str = "t5-small",
        dataset_names: list = ["din0s/ccmatrix_en-ro", "j0hngou/ccmatrix_en-fr", "j0hngou/ccmatrix_de-en"],
        source_target_pair: list = [("en", "ro"), ("en", "fr"), ("en", "de")],
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        group_pairs: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Make sure the tokenizer and the dataset are downloaded
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        for dataset in self.hparams.dataset_names:
            load_dataset(dataset, use_auth_token=True)

    def setup(self, stage: Optional[str] = None):
        # Create a list with all the datasets
        datasets = [MTDistillationDataset(name, pair[0], pair[1])
                    for pair, name in zip(self.hparams.source_target_pair, self.hparams.dataset_names)]

        datasets = self.split_dataset(datasets)

        self.dataset = {}
        self.dataset['train'] = ConcatDataset(datasets['train'])
        self.dataset['val'] = ConcatDataset(datasets['val'])
        self.dataset['test'] = ConcatDataset(datasets['test'])

        # Create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

    @staticmethod
    def split_dataset(datasets: List[Dataset]) -> dict[str, List[Dataset]]:
        train_list = []
        val_list = []
        test_list = []
        for dataset in datasets:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                                  len(dataset) - int(
                                                                                      len(dataset) * 0.8)])
            test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [int(len(test_dataset) * 0.5),
                                                                                     len(test_dataset) - int(
                                                                                         len(test_dataset) * 0.5)])

            train_list.append(train_dataset)
            val_list.append(val_dataset)
            test_list.append(test_dataset)

        return {'train': train_list, 'val': val_list, 'test': test_list}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset['train'],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_group if self.hparams.group_pairs else self.collate_fn,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset['val'],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_group if self.hparams.group_pairs else self.collate_fn,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset['test'],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_group if self.hparams.group_pairs else self.collate_fn,
            shuffle=False,
            drop_last=True
        )

    def collate_fn_group(self, batch: list[dict[str, str]]):
        sentences = {}

        # Tokenize
        source = self.tokenizer([sentence[0] for sentence in batch], return_tensors="pt", padding=True)
        target = self.tokenizer([sentence[1] for sentence in batch], return_tensors="pt", padding=True)

        # Group sentences by language pairs
        pairs_in_batch = [sample[2:] for sample in batch]
        for pair in self.hparams.source_target_pair:
            # Skip pairs with less than two samples in the batch
            if pairs_in_batch.count(pair) < 2:
                continue
            # Get the samples that belong to the current pair
            sentences[f"{pair[0]}-{pair[1]}"] = {}
            sentences[f"{pair[0]}-{pair[1]}"]['input_ids'] = torch.stack([source.input_ids[i]
                                                                          for i, sample in enumerate(batch)
                                                                          if tuple(sample[2:]) == pair])
            sentences[f"{pair[0]}-{pair[1]}"]['attention_mask'] = torch.stack([source.attention_mask[i]
                                                                               for i, sample in enumerate(batch)
                                                                               if tuple(sample[2:]) == pair])
            sentences[f"{pair[0]}-{pair[1]}"]['decoder_input_ids'] = torch.stack([target.input_ids[i]
                                                                                  for i, sample in enumerate(batch)
                                                                                  if tuple(sample[2:]) == pair])
            sentences[f"{pair[0]}-{pair[1]}"]['decoder_attention_mask'] = torch.stack([target.attention_mask[i]
                                                                                       for i, sample in enumerate(batch)
                                                                                       if tuple(sample[2:]) == pair])

        return sentences

    def collate_fn(self, batch: list[dict[str, str]]):
        sentences = {}

        # Tokenize
        source = self.tokenizer([sentence[0] for sentence in batch], return_tensors="pt", padding=True)
        target = self.tokenizer([sentence[1] for sentence in batch], return_tensors="pt", padding=True)

        sentences['input_ids'] = source.input_ids
        sentences['attention_mask'] = source.attention_mask
        sentences['decoder_input_ids'] = target.input_ids
        sentences['decoder_attention_mask'] = target.attention_mask

        return sentences


class MTDistillationDataset(Dataset):
    def __init__(self, dataset_name, source_lang, target_lang):

        self.dataset = load_dataset(dataset_name, use_auth_token=True)["train"]
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.process_data()

    def __len__(self):
        return len(self.dataset)

    def process_data(self):
        self.dataset = self.dataset.map(
            self.prep_ccmatrix,
            batched=True,
            remove_columns=self.dataset.column_names,
        )

    @staticmethod
    def prep_ccmatrix(batch: dict[str, list]) -> dict[str, list]:
        new_batch = {}

        pairs = batch["translation"]
        for lang in pairs[0].keys():
            new_batch[lang] = [pair[lang] for pair in pairs]

        return new_batch

    def __getitem__(self, idx):
        src_lang = self.get_full_lang_name(self.source_lang)
        tgt_lang = self.get_full_lang_name(self.target_lang)
        prefix = f"translate {src_lang} to {tgt_lang}: "

        source = prefix + self.dataset[idx][self.source_lang]
        target = self.dataset[idx][self.target_lang]

        return source, target, self.source_lang, self.target_lang

    @staticmethod
    def get_full_lang_name(lang):
        if lang == "en":
            return "English"
        elif lang == "ro":
            return "Romanian"
        elif lang == "de":
            return "German"
        elif lang == "fr":
            return "French"
        elif lang == "it":
            return "Italian"
        else:
            raise ValueError(f"Language {lang} not supported")
