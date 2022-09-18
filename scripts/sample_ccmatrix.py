from datasets import load_dataset

import numpy as np

HF_USER = "din0s"
TAG = "en-ro"

if __name__ == "__main__":
    dataset = load_dataset("yhavinga/ccmatrix", TAG)
    n_samples_orig = dataset["train"].num_rows
    n_samples_new = 1_000_000

    np.random.seed(42)
    idx = np.random.choice(n_samples_orig, (n_samples_new,))
    dataset["train"].select(idx).push_to_hub(f"{HF_USER}/ccmatrix_{TAG}", private=True)
