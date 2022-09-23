import transformers
from datasets import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
import datasets
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--language_name', type=str, default='Romanian')
parser.add_argument('--code', type=str, default='ro')
parser.add_argument('--splits', nargs=3, type=int, default=[1500, 3000, 12000])
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()


model_checkpoint = "t5-small"
language = {}
language['name'] = args.language_name
language['code'] = args.code

dataset = load_dataset(f"j0hngou/ccmatrix_en-{language['code']}")

low_resource_dev = load_dataset(path=f"j0hngou/ccmatrix_en-{language['code']}", split=f"train[:{args.splits[0]}]")
low_resource_test = load_dataset(path=f"j0hngou/ccmatrix_en-{language['code']}", split=f"train[{args.splits[0]}:{args.splits[1]}]")
low_resource_train = load_dataset(path=f"j0hngou/ccmatrix_en-{language['code']}", split=f"train[{args.splits[1]}:{args.splits[2]}]")

raw_datasets = DatasetDict({ 
  "train": low_resource_train,
  "validation": low_resource_dev,
  "test": low_resource_test
})

metric = load_metric("sacrebleu")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if "mbart" in model_checkpoint:
    tokenizer.src_lang = "en-ro"
    tokenizer.tgt_lang = f"en-{language['code']}"

prefix = f"translate English to {language['name']}: "

max_input_length = 256
max_target_length = 256
source_lang = "en"
target_lang = "it"

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.config.max_length = 256

batch_size = args.batch_size
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=args.num_epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    report_to='wandb',
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub()
