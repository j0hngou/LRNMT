import sys
import argparse
import tqdm
import torch

sys.path.append('../')

from transformers import AutoModelForSeq2SeqLM
from datamodules import MTDistillationDatamodule
from transformers import AutoTokenizer

def preprocessString(string):
    # escape double quotes
    string = string.replace('"', '\\"')
    # remove prefix
    prefix = "translate Italian to English: "
    string = string.replace(prefix, '')
    return string


def generate_synthetic_data(model, dm, save_location, num_samples=1000):
    num_generated = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = dm.train_dataloader().batch_size
    model.to(device)
    for batch in tqdm.tqdm(dm.train_dataloader(), total=num_samples // batch_size):
        input_ids = batch['it-en']['input_ids'].to(device)
        attention_mask = batch['it-en']['attention_mask'].to(device)
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=256, num_beams=5, early_stopping=True)
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        decoded_input = tokenizer.batch_decode(batch['it-en']['input_ids'], skip_special_tokens=True)
        with open(save_location, 'a', encoding='utf-8') as f:
            for line in zip(decoded_input, decoded):
                f.write(f'{{ "it": "{preprocessString(line[0])}", "en": "{preprocessString(line[1])}" }}\n')
        num_generated += len(decoded)
        if num_generated >= num_samples:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='din0s/t5-small-ro-finetuned-en-to-it')
    parser.add_argument('--dataset_name', type=str, default='j0hngou/ccmatrix_en-it')
    parser.add_argument('--save_location', type=str, default=f'data/synthetic_data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--start_dataset', type=int, default=15000)
    parser.add_argument('--end_dataset', type=int, default=27000)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    save_location = args.save_location
    batch_size = args.batch_size
    num_samples = args.num_samples

    splits = [[0, 1500], [1500, 3000], [args.start_dataset, args.end_dataset]]
    dm = MTDistillationDatamodule(dataset_names=[dataset_name], source_target_pair=[('it', 'en')],
                                  batch_size=batch_size, splits=splits)
    dm.setup()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    save_location = f'{save_location}_{model_name.split("/")[-1]}.jsonl'

    generate_synthetic_data(model, dm, save_location, num_samples)
