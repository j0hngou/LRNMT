from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl


class MTDistillationDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_names: list = ["Helsinki-NLP/opus-mt-en-ro", "Helsinki-NLP/opus-mt-en-fr",],
        dataset_names: list = ["din0s/ccmatrix_en-ro", "j0hngou/ccmatrix_en-fr"],
        source_target_pair: list = [("en", "ro"), ("en", "fr")],
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Make sure the tokenizer and the dataset are downloaded
        for tokenizer in self.hparams.tokenizer_names:
            AutoTokenizer.from_pretrained(tokenizer)
        for dataset in self.hparams.dataset_names:
            load_dataset(dataset, use_auth_token=True)

    def setup(self, stage: Optional[str] = None):
        # TODO: split
        # Create a list with all the datasets and then concat them
        datasets = [MTDistillationDataset(name, pair[0], pair[1], preappend_tg_lang=True)
                    for pair, name in zip(self.hparams.source_target_pair, self.hparams.dataset_names)]
        self.dataset = ConcatDataset(datasets)

        # Create a tokenizer for each pair including the multilingual one
        self.tokenizers = {}
        for pair, tokenizer in zip(self.hparams.source_target_pair, self.hparams.tokenizer_names):
            self.tokenizers[f"{pair[0]}-{pair[1]}"] = AutoTokenizer.from_pretrained(tokenizer)

        # TODO: need to create a new one with the whole vocab and the pre-appended token for the target language
        self.multiling_tokenizer = self.tokenizers["en-ro"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def collate_fn(self, batch: list[dict[str, str]]):
        biling_sentences = {}
        multiling_sentence = {}

        # Group pairs for tokenization
        for pair in self.hparams.source_target_pair:
            biling_sentences[f"{pair[0]}-{pair[1]}"] = [sample[:2] for sample in batch if tuple(sample[2:]) == pair]
            # same as above but pre-append target language identification token
            multiling_sentence[f"{pair[0]}-{pair[1]}"] = [(f"<{pair[1]}> " + sample[0], sample[1]) for sample in batch
                                                          if tuple(sample[2:]) == pair]

        # Tokenize each language from each pair separately
        biling_ids = {}
        multiling_ids = {}
        for pair in biling_sentences.keys():
            # Skip pairs with less than 2 samples
            if len(biling_sentences[pair]) < 2:
                continue
            biling_ids[pair] = {
                "source": self.tokenizers[pair]([sentence[0] for sentence in biling_sentences[pair]], return_tensors="pt", padding=True).input_ids,
                "target": self.tokenizers[pair]([sentence[1] for sentence in biling_sentences[pair]], return_tensors="pt", padding=True).input_ids,
            }
            # TODO: not sure if we need the target_ids from the multilingual or the bilingial tokenizer for the multilingual model
            # TODO: TO-REVISIT
            multiling_ids[pair] = {
                "source": self.multiling_tokenizer([sentnece[0] for sentnece in multiling_sentence[pair]], return_tensors="pt", padding=True).input_ids,
                "target": self.multiling_tokenizer([sentence[1] for sentence in multiling_sentence[pair]], return_tensors="pt", padding=True).input_ids,
            }

        return {"biling_ids": biling_ids, "multiling_ids": multiling_ids}


class MTDistillationDataset(Dataset):
    def __init__(self, dataset_name, source_lang, target_lang, preappend_tg_lang=True):

        self.dataset = load_dataset(dataset_name, use_auth_token=True)["train"]
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.preappend_tg_lang = preappend_tg_lang

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
        source = self.dataset[idx][self.source_lang]
        target = self.dataset[idx][self.target_lang]
        return source, target, self.source_lang, self.target_lang
