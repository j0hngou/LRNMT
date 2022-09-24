from Distiller import Distiller
import transformers
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import sys
sys.path.append('../')
from datamodules import MTDistillationDatamodule
from datasets import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
import argparse
import wandb
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="distiller", entity="deeplearning2")

teacher_checkpoint = "t5-base"

parser = argparse.ArgumentParser()

metric = load_metric("sacrebleu")

tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint)

language_codes = ['en-ro', 'en-fr']

max_input_length = 256
max_target_length = 256

batch_size = 2

dm = MTDistillationDatamodule(batch_size=batch_size, group_pairs=False)
dm.setup()

distiller = Distiller(
    teacher=AutoModelForSeq2SeqLM.from_pretrained(teacher_checkpoint),
    n=3,
    temperature=1,
    loss_weights=[1/2, 1/2, 0],
)



trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
    logger=wandb_logger,
    # precision=16, # lisa fp16
    # amp_backend="apex", # lisa fp16
    # amp_level="O2", # lisa fp16
)

trainer.fit(distiller, dm)