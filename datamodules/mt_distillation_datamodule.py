from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl


class MTDistillationDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str = "Helsinki-NLP/opus-mt-en-ro",
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
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        for dataset in self.hparams.dataset_names:
            load_dataset(dataset, use_auth_token=True)

    def setup(self, stage: Optional[str] = None):
        # TODO: split
        # Create a list with all the datasets and then concat them
        datasets = [MTDistillationDataset(name, pair[0], pair[1], preappend_tg_lang=True)
                    for pair, name in zip(self.hparams.source_target_pair, self.hparams.dataset_names)]
        self.dataset = ConcatDataset(datasets)

        # Create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def collate_fn(self, batch: list[dict[str, str]]):
        sentences = {}

        # Tokenize
        source_ids = self.tokenizer([sentence[0] for sentence in batch], return_tensors="pt", padding=True)
        target_ids = self.tokenizer([sentence[1] for sentence in batch], return_tensors="pt", padding=True).input_ids

        # Group sentences by language pairs
        for pair in self.hparams.source_target_pair:
            sentences[f"{pair[0]}-{pair[1]}"] = {}
            sentences[f"{pair[0]}-{pair[1]}"]['source'] = [source_ids.input_ids[i] for i, sample in enumerate(batch)
                                                           if tuple(sample[2:]) == pair]
            sentences[f"{pair[0]}-{pair[1]}"]['attention_mask'] = [source_ids.attention_mask[i] for i, sample in enumerate(batch)
                                                           if tuple(sample[2:]) == pair]
            sentences[f"{pair[0]}-{pair[1]}"]['target'] = [target_ids[i] for i, sample in enumerate(batch)
                                                           if tuple(sample[2:]) == pair]

        return sentences


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
