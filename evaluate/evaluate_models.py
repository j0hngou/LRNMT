from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate
import numpy as np
import os
import json

BATCH_SIZE = 32
WITH_BUCKETS = True
SAVE_DIR = "eval_results"
BUCKET_SIZES = [32, 64, 128, 256, 512]

datasets = {
    "en-it": "j0hngou/ccmatrix_en-it",
    "en-ro": "din0s/ccmatrix_en-ro",
    "en-fr": "j0hngou/ccmatrix_en-fr",
}

languages = {
    "French": "fr",
    "Italian": "it",
    "Romanian": "ro",
    "English": "en",
}

models = {
    "base": "t5-base",
    "en-fr": "j0hngou/t5-base-finetuned-en-to-fr",
    "en-ro": "j0hngou/t5-base-finetuned-en-to-ro",
    "en-it": "din0s/t5-base-finetuned-en-to-it",
    "en-fr-it": "din0s/t5-base_fr-finetuned-en-to-it",
    "en-ro-it": "din0s/t5-base_ro-finetuned-en-to-it",
    "hrs": "din0s/t5-base-finetuned-en-to-it-hrs",
    "lrs": "din0s/t5-base-finetuned-en-to-it-lrs",
}

model_data_dict = {
    # ("French", models["base"]): datasets["en-fr"],
    # ("Romanian", models["base"]): datasets["en-ro"],
    # ("Italian", models["base"]): datasets["en-it"],
    # ("French", models["en-fr"]): datasets["en-fr"],
    # ("Romanian", models["en-ro"]): datasets["en-ro"],
    ("Italian", models["en-it"]): datasets["en-it"],
    ("Italian", models["en-fr-it"]): datasets["en-it"],
    ("Italian", models["en-ro-it"]): datasets["en-it"],
    ("Italian", models["hrs"]): datasets["en-it"],
    ("Italian", models["lrs"]): datasets["en-it"],
}

# Caution: assuming all models have the same tokenizer
tokenizer = AutoTokenizer.from_pretrained(models["base"], model_max_length=512)
metric = evaluate.load("sacrebleu")
max_input_length = 256
max_target_length = 256


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


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

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def do_eval(model, train_args, tokenized_test_set, data_collator, tokenizer):
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        eval_dataset=tokenized_test_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer.evaluate(num_beams=5, max_length=max_target_length)


os.makedirs(SAVE_DIR, exist_ok=True)
for (lang_name, model_name), dataset in model_data_dict.items():
    # Load test split
    # TODO: wrong split for high resource
    test_set = load_dataset(path=dataset, split="train[1500:3000]")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    source_lang = "en"
    target_lang = languages[lang_name]
    prefix = f"translate English to {lang_name}: "

    # Remove user if present in model name
    model_name = model_name.split("/")[-1]
    output_dir = f"{SAVE_DIR}/{model_name}-{target_lang}"

    args = Seq2SeqTrainingArguments(
        output_dir,
        generation_max_length=max_target_length,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        report_to="none",
    )

    if WITH_BUCKETS:
        prev_bucket = 0
        for bucket in BUCKET_SIZES:
            test_set_i = test_set.filter(
                lambda x: len(x["translation"]["en"]) <= bucket
                and len(x["translation"]["en"]) > prev_bucket
            )
            prev_bucket = bucket

            tokenized_test_set_i = test_set_i.map(preprocess_function, batched=True)
            metrics = do_eval(model, args, tokenized_test_set_i, data_collator, tokenizer)
            with open(f"{output_dir}/metrics-{bucket}.json", "w") as f:
                json.dump(metrics, f)
    else:
        tokenized_test_set = test_set.map(preprocess_function, batched=True)
        metrics = do_eval(model, args, tokenized_test_set, data_collator, tokenizer)
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f)
