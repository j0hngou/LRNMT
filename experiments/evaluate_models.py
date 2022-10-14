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
import argparse

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
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "en-fr": "j0hngou/t5-base-finetuned-en-to-fr",
    "en-ro": "j0hngou/t5-base-finetuned-en-to-ro",
    "en-it": "din0s/t5-base-finetuned-en-to-it",
    "en-it-small": "din0s/t5-small-finetuned-en-to-it",
    "en-fr-it": "din0s/t5-base_fr-finetuned-en-to-it",
    "en-fr-it-small": "din0s/t5-small-fr-finetuned-en-to-it",
    "en-ro-it": "din0s/t5-base_ro-finetuned-en-to-it",
    "en-ro-it-small": "din0s/t5-small-ro-finetuned-en-to-it",
    "hrs-small": "din0s/t5-small-finetuned-en-to-it-hrs",
    "hrs-base": "din0s/t5-base-finetuned-en-to-it-hrs",
    "lrs-small": "din0s/t5-small-finetuned-en-to-it-lrs",
    "lrs-base": "din0s/t5-base-finetuned-en-to-it-lrs",
    "lrs-small-back": "din0s/t5-small-finetuned-en-to-it-lrs-back",
    "lrs-base-back": "din0s/t5-base-finetuned-en-to-it-lrs-back",
    "dual-kd-back": "j0hngou/2teachersdistillbacktranslation-en-it",
}

bar_plot_model_data_dict = {
    ("Italian", models["en-it"]): datasets["en-it"],
    ("Italian", models["en-it-small"]): datasets["en-it"],
    ("Italian", models["lrs-base-back"]): datasets["en-it"],
    ("Italian", models["lrs-small-back"]): datasets["en-it"],
    ("Italian", models["dual-kd-back"]): datasets["en-it"],
}

table_model_data_dict = {
    # No fine-tuning
    ("Italian", models["t5-small"]): datasets["en-it"],
    ("Italian", models["t5-base"]): datasets["en-it"],
    # Fine-tuning on Italian
    ("Italian", models["en-it-small"]): datasets["en-it"],
    ("Italian", models["en-it"]): datasets["en-it"],
    ("Italian", models["en-fr-it-small"]): datasets["en-it"],
    ("Italian", models["en-fr-it"]): datasets["en-it"],
    ("Italian", models["en-ro-it-small"]): datasets["en-it"],
    ("Italian", models["en-ro-it"]): datasets["en-it"],
    # Fine-tuning w/ data augmentation
    ("Italian", models["hrs-small"]): datasets["en-it"],
    ("Italian", models["hrs-base"]): datasets["en-it"],
    ("Italian", models["lrs-small"]): datasets["en-it"],
    ("Italian", models["lrs-base"]): datasets["en-it"],
    ("Italian", models["lrs-small-back"]): datasets["en-it"],
    ("Italian", models["lrs-base-back"]): datasets["en-it"],
    # Knowledge distillation
    ("Italian", models["dual-kd-back"]): datasets["en-it"],
}


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_type', type=str, default='bar_plot', choices=['bar_plot', 'table'])
    parser.add_argument('--save_dir', type=str, default='eval_results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--bucket_sizes', type=list, default=[64, 128, 512])

    args = parser.parse_args()

    model_data_dict = bar_plot_model_data_dict if args.result_type == 'bar_plot' else table_model_data_dict
    save_dir = args.save_dir
    batch_size = args.batch_size
    with_buckets = True if args.result_type == 'bar_plot' else False
    bucket_sizes = args.bucket_sizes

    # Caution: assuming all models have the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(models["t5-base"], model_max_length=512)
    metric = evaluate.load("sacrebleu")
    max_input_length = 256
    max_target_length = 256

    os.makedirs(save_dir, exist_ok=True)
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
        output_dir = f"{save_dir}/{model_name}-{target_lang}"

        args = Seq2SeqTrainingArguments(
            output_dir,
            generation_max_length=max_target_length,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            report_to="none",
        )

        if with_buckets:
            prev_bucket = 0
            for bucket in bucket_sizes:
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
