from datasets import load_dataset
from transformers import AutoTokenizer

path = {"it": "j0hngou/ccmatrix_en-it", "fr": "j0hngou/ccmatrix_en-fr", "ro": "din0s/ccmatrix_en-ro"}

# Load the datasets
dataset = {}
for key, value in path.items():
    if key == "it":
        dataset[key] = load_dataset(value, split="train[3000:15000]")["translation"]
    else:
        dataset[key] = load_dataset(value)["train"]["translation"]

# Keep only the Italian, French and Romanian sentences
sentences = {}
for key, value in dataset.items():
    sentences[key] = []
    for pair in value:
        sentences[key].append(pair[key])

del dataset

# Calculate the average length of the sentences
avg_len = {}
for key, value in sentences.items():
    avg_len[key] = sum([len(sentence.split()) for sentence in value]) / len(value)

tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenize the sentences
tokenized_sentences = {}
batch_size = 1000
for key, value in sentences.items():
    tokens = []
    for i in range(0, len(value), batch_size):
        tokens.extend(tokenizer(value[i: i + batch_size]).input_ids)
    tokenized_sentences[key] = tokens

# Get the unigrams, bigrams and trigrams
unigrams = {}
bigrams = {}
trigrams = {}
for key, value in tokenized_sentences.items():
    unigrams[key] = set()
    bigrams[key] = set()
    trigrams[key] = set()
    for sentence in value:
        for i, token in enumerate(sentence):
            unigrams[key].add(token)
            if i < len(sentence) - 1:
                bigrams[key].add((token, sentence[i + 1]))
            if i < len(sentence) - 2:
                trigrams[key].add((token, sentence[i + 1], sentence[i + 2]))

# Calculate Intersections of unigrams, bigrams and trigrams
unigrams_intersection = {"it-fr": unigrams["it"].intersection(unigrams["fr"]),
                         "it-ro": unigrams["it"].intersection(unigrams["ro"]),
                         "fr-ro": unigrams["fr"].intersection(unigrams["ro"])}
bigrams_intersection = {"it-fr": bigrams["it"].intersection(bigrams["fr"]),
                        "it-ro": bigrams["it"].intersection(bigrams["ro"]),
                        "fr-ro": bigrams["fr"].intersection(bigrams["ro"])}
trigrams_intersection = {"it-fr": trigrams["it"].intersection(trigrams["fr"]),
                         "it-ro": trigrams["it"].intersection(trigrams["ro"]),
                         "fr-ro": trigrams["fr"].intersection(trigrams["ro"])}

# Print the intersections and their difference
print("Unigram intersetion between Italian and French: ", len(unigrams_intersection["it-fr"]))
print("Unigram intersetion between Italian and Romanian: ", len(unigrams_intersection["it-ro"]))
print("Unigram intersetion between French and Romanian: ", len(unigrams_intersection["fr-ro"]))
print("Unigram it-fr diff it-ro", len(unigrams_intersection["it-fr"] - unigrams_intersection["it-ro"]))
print("Unigram it-ro diff it-fr", len(unigrams_intersection["it-ro"] - unigrams_intersection["it-fr"]))
print("Unigram (itro inters itft) - frro", len(unigrams_intersection["it-ro"].union(unigrams_intersection["it-fr"]) - unigrams_intersection["fr-ro"]))
print()
print("Bigram intersetion between Italian and French: ", len(bigrams_intersection["it-fr"]))
print("Bigram intersetion between Italian and Romanian: ", len(bigrams_intersection["it-ro"]))
print("Bigram intersetion between French and Romanian: ", len(bigrams_intersection["fr-ro"]))
print("Bigram it-fr diff it-ro", len(bigrams_intersection["it-fr"] - bigrams_intersection["it-ro"]))
print("Bigram it-ro diff it-fr", len(bigrams_intersection["it-ro"] - bigrams_intersection["it-fr"]))
print("Bigram (itro inters itft) - frro",len(bigrams_intersection["it-ro"].union(bigrams_intersection["it-fr"]) - bigrams_intersection["fr-ro"]))
print()
print("Trigram intersetion between Italian and French: ", len(trigrams_intersection["it-fr"]))
print("Trigram intersetion between Italian and Romanian: ", len(trigrams_intersection["it-ro"]))
print("Trigram intersetion between French and Romanian: ", len(trigrams_intersection["fr-ro"]))
print("Trigram it-fr diff it-ro", len(trigrams_intersection["it-fr"] - trigrams_intersection["it-ro"]))
print("Trigram it-ro diff it-fr", len(trigrams_intersection["it-ro"] - trigrams_intersection["it-fr"]))
print("Trigram (itro inters itft) - frro", len(trigrams_intersection["it-ro"].union(trigrams_intersection["it-fr"]) - trigrams_intersection["fr-ro"]))
print()
# Print the length of unigrams, bigrams and trigrams
print(f"Unigrams Length: It {len(unigrams['it'])}, Fr {len(unigrams['fr'])}, Ro {len(unigrams['ro'])}")
print(f"Bigrams Length: It {len(bigrams['it'])}, Fr {len(bigrams['fr'])}, Ro {len(bigrams['ro'])}")
print(f"Trigrams Length: It {len(trigrams['it'])}, Fr {len(trigrams['fr'])}, Ro {len(trigrams['ro'])}")
print()
# Print average length of sentences
print("Average length of Italian sentences: ", avg_len["it"])
print("Average length of French sentences: ", avg_len["fr"])
print("Average length of Romanian sentences: ", avg_len["ro"])
