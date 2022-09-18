from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional

import pytorch_lightning as pl


class CCMatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def prepare_data(self):
        # Make sure the tokenizer and the dataset are downloaded
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        load_dataset(self.hparams.dataset_name, use_auth_token=True)

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset(self.hparams.dataset_name, use_auth_token=True)

        for split in self.dataset.keys():
            # Reduce the dataset's features to just the sentences in the source & target languages
            self.dataset[split] = self.dataset[split].map(
                self.prep_ccmatrix,
                batched=True,
                remove_columns=self.dataset[split].column_names,
            )

    def prep_ccmatrix(self, batch: dict[str, list]) -> dict[str, list]:
        new_batch = {}

        pairs = batch["translation"]
        for lang in pairs[0].keys():
            new_batch[lang] = [pair[lang] for pair in pairs]

        return new_batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def collate_fn(self, batch: list[dict[str, str]]) -> dict[str, Tensor]:
        sentences = {}

        langs = batch[0].keys()
        for lang in langs:
            # Batch tokenized sentences per language (with padding to max length in batch)
            sentences[lang] = self.tokenizer(
                [pair[lang] for pair in batch], padding=True, return_tensors="pt"
            ).input_ids

        return sentences
