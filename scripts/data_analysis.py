from datasets import load_dataset
from transformers import AutoTokenizer, T5TokenizerFast
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2


def get_dataset(path: dict, splits: list) -> dict:
    dataset = {}
    for key, split in zip(path.keys(), splits):
        dataset[key] = load_dataset(path[key], split=f"train[{split[0]}:{split[1]}]")["translation"]
    return dataset


def get_sentences(dataset:dict, languages: list) -> dict:
    sentences = {}
    for key, lang in zip(dataset.keys(), languages):
        sentences[key] = []
        for pair in dataset[key]:
            sentences[key].append(pair[lang])

    return sentences


def get_avg_len(sentences: dict) -> dict:
    avg_len = {}
    for key, value in sentences.items():
        avg_len[key] = sum([len(sentence.split()) for sentence in value]) / len(value)
    return avg_len


def tokenize_sentences(sentences: dict, tokenizer: T5TokenizerFast, batch_size=1000) -> dict:
    tokenized_sentences = {}
    for key, value in sentences.items():
        tokens = []
        for i in range(0, len(value), batch_size):
            tokens.extend(tokenizer(value[i: i + batch_size]).input_ids)
        tokenized_sentences[key] = tokens

    return tokenized_sentences


def get_ngrams(tokenized_sentences: dict) -> tuple[dict, dict, dict]:
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
    return unigrams, bigrams, trigrams


def get_intersection(ngrams: dict):
    intersection = {}
    keys = list(ngrams.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            intersection[f"{keys[i]}-{keys[j]}"] = ngrams[keys[i]].intersection(ngrams[keys[j]])
    return intersection


def pretty_print(intersection: dict, ngram: dict, ngram_name: str):
    keys = list(intersection.keys())
    print(f"Analysis of {ngram_name}")
    print('*'*50)
    # Print the length of the intersections
    for key in keys:
        print(f"{ngram_name} intersection between {key}: {len(intersection[key])}")

    print('-'*50)
    # Print the length of the difference between the intersections
    for key in keys:
        keys_ = keys.copy()
        keys_.remove(key)
        for key_ in keys_:
            print(f"{ngram_name} difference between {key} and {key_}: "
                  f"{len(intersection[key].difference(intersection[key_]))}")

    print('-'*50)
    # Print the length of the difference of the union between two intersections and the other
    for key in keys:
        keys_ = keys.copy()
        keys_.remove(key)
        print(f"{ngram_name} the union of  {keys_[0]} and {keys_[1]} diff {key}  : "
                f"{len(intersection[keys_[0]].union(intersection[keys_[1]]).difference(intersection[key]))}")

    print('-'*50)
    # Print the length of the ngrams
    for key in ngram.keys():
        print(f"{ngram_name} length {key} : {len(ngram[key])}")

    print('*'*50)


def plot_venn(path: dict, splits: list, languages: list, save: bool, name: str):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    dataset = get_dataset(path, splits)
    sentences = get_sentences(dataset, languages)
    tokenized_sentences = tokenize_sentences(sentences, tokenizer)
    unigrams, bigrams, trigrams = get_ngrams(tokenized_sentences)

    keys = list(path.keys())
    for i, ngram in enumerate([unigrams, bigrams, trigrams]):
        ngram_name = ["unigrams", "bigrams", "trigrams"][i]
        if len(keys) == 2:
            venn2([ngram[keys[0]], ngram[keys[1]]], keys)
        elif len(keys) == 3:
            venn3([ngram[keys[0]], ngram[keys[1]], ngram[keys[2]]], keys)

        if save:
            plt.savefig(f"venn_{ngram_name}_{name}.png")
            plt.clf()
        else:
            plt.show()


def print_stats(path: dict, splits: list, languages: list):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    dataset = get_dataset(path, splits)
    sentences = get_sentences(dataset, languages)
    avg_len = get_avg_len(sentences)
    tokenized_sentences = tokenize_sentences(sentences, tokenizer)
    unigrams, bigrams, trigrams = get_ngrams(tokenized_sentences)

    unigrams_intersection = get_intersection(unigrams)
    bigrams_intersection = get_intersection(bigrams)
    trigrams_intersection = get_intersection(trigrams)

    for key in avg_len.keys():
        print(f"Average length of {key} sentences: {avg_len[key]}")
    print('*' * 50)

    pretty_print(unigrams_intersection, unigrams, "Unigrams")
    pretty_print(bigrams_intersection, bigrams, "Bigrams")
    pretty_print(trigrams_intersection, trigrams, "Trigrams")


if __name__ == "__main__":
    # path = {"it": "j0hngou/ccmatrix_en-it", "fr": "j0hngou/ccmatrix_en-fr", "ro": "din0s/ccmatrix_en-ro"}
    path = {"it": "j0hngou/ccmatrix_en-it", "de": "j0hngou/ccmatrix_de-en"}
    # path = {"it": "j0hngou/ccmatrix_en-it", "ithrs": "irenepap/en-it-hrs-data", "itlrs": "irenepap/en-it-lrs-data"}

    # splits = [("", ""), ("12000", ""), ("12000", "")]
    # splits = [("", ""), ("", ""), ("", "")]
    splits = [("", ""), ("", "")]
    languages = ["it", "de"]

    # print_stats(path, splits, languages)
    plot_venn(path, splits, languages, save=True, name='IT-DE')

