from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

import torch

def perplexity(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dataset_split: Dataset,
    language: str = 'it',
    stride: int = 512,
) -> float:
    """
    Compute perplexity of a model on a dataset split.
    https://huggingface.co/docs/transformers/perplexity

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        dataset_split: The dataset split to evaluate on.
        language: The language to calculate the perplexity on.
        stride: The stride to use for the sliding window.
    """
    prev_device = model.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = [x['translation'][language] for x in dataset_split]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")

    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    model.to(prev_device)

    return ppl.item()
