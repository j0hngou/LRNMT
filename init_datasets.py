from datasets import load_dataset

# Load the English-German dataset, the English-French dataset, and the English-Romanian dataset, and the English-Italian dataset

en_de = load_dataset("yhavinga/ccmatrix", "en-de")
en_fr = load_dataset("yhavinga/ccmatrix", "en-fr")
en_ro = load_dataset("yhavinga/ccmatrix", "en-ro")
en_it = load_dataset("yhavinga/ccmatrix", "en-it")

dataset = load_dataset("yhavinga/ccmatrix", config="en-fr")