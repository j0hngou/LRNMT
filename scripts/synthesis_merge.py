from datasets import concatenate_datasets, load_dataset, Features, Translation

# Define the mapping for the translation feature
translation_features = Features({
    "translation": Translation(languages=["en", "it"]),
})

# Load the datasets from HuggingFace
synth_fr = load_dataset("irenepap/en-fr2it-synthetic-data", split='train', features=translation_features)
synth_ro = load_dataset("irenepap/en-ro2it-synthetic-data", split='train', features=translation_features)
synth_it = load_dataset("irenepap/en-it-synthetic-data", split='train', features=translation_features)
original = load_dataset("j0hngou/ccmatrix_en-it_subsampled", split='train').remove_columns(["id", "score"])

# Select from the original samples according to the size of the synthetic data
train_idx_start = 3000
train_idx_end = train_idx_start + len(synth_fr)
original = original.select(range(train_idx_start, train_idx_end))

# Concatenate either the high or low resource synthetic data with the original
hrs_data = concatenate_datasets([original, synth_fr, synth_ro])
lrs_data = concatenate_datasets([original, synth_it])
assert hrs_data.num_rows == lrs_data.num_rows, "HRS and LRS datasets should have the same size"

# Push the merged datasets to the HuggingFace hub
lrs_data.push_to_hub("en-it-lrs-data")
hrs_data.push_to_hub("en-it-hrs-data")
