from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
import numpy as np
import json

it_dataset = "j0hngou/ccmatrix_en-it"

languages = {"French" : "fr", "German" : "de", "Italian" : "it", "Romanian" : "ro", "English" : "en"}

model_data_dict = {
    ("French", "din0s/t5-small-finetuned-en-to-fr") : "j0hngou/ccmatrix_en-fr", 
    ("Romanian", "din0s/t5-small-finetuned-en-to-ro") : "din0s/ccmatrix_en-ro", 
    ("German","din0s/t5-small-finetuned-en-to-de"): "j0hngou/ccmatrix_de-en",
    ("Italian", "din0s/t5-small-fr-finetuned-en-to-it") : it_dataset, 
    ("Italian", "din0s/t5-small-ro-finetuned-en-to-it") : it_dataset,
    ("Italian", "din0s/t5-small-de-finetuned-en-to-it") : it_dataset,
    ("Italian", "din0s/t5-small-finetuned-en-to-it") : it_dataset,
    ("French", "t5-small") : "j0hngou/ccmatrix_en-fr", 
    ("Romanian", "t5-small") : "din0s/ccmatrix_en-ro", 
    ("German", "t5-small"): "j0hngou/ccmatrix_de-en",
    ("Italian", "t5-small") : it_dataset,
}

tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
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

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


for (lang_name, model_name), dataset in model_data_dict.items():

    test_set = load_dataset(path=dataset, split="train[1500:3000]")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    source_lang = "en"
    target_lang = languages[lang_name]
    prefix = f"translate English to {lang_name}: "
    tokenized_test_set = test_set.map(preprocess_function, batched=True)

    if len(model_name.split('/')) == 1:
        model_name = model_name.split('/')[0]
    else:
        model_name = model_name.split('/')[1]

    args = Seq2SeqTrainingArguments(
        f"{model_name}-{languages[lang_name]}", 
        generation_max_length=max_target_length,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        eval_dataset=tokenized_test_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate(num_beams=5, max_length=max_target_length)

    #save dict to json
    with open(f"eval_results/{model_name}_{languages[lang_name]}_metrics.json", "w") as f:
        json.dump(metrics, f)