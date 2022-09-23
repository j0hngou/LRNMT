from datasets import load_dataset

import numpy as np

HF_USER = "j0hngou"
TAG = "en-it"

if __name__ == "__main__":
    dataset = load_dataset('j0hngou/ccmatrix_en-it')
    n_samples_orig = dataset["train"].num_rows
    n_samples_new = 15000

    np.random.seed(42)
    idx = np.random.choice(n_samples_orig, (n_samples_new,))
    dataset["train"].select(idx).push_to_hub(f"{HF_USER}/ccmatrix_{TAG}_subsampled", private=False)
