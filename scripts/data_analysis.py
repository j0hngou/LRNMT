from datasets import load_dataset
from transformers import AutoTokenizer, T5TokenizerFast
from matplotlib_venn import venn3, venn2

import matplotlib.pyplot as plt


def get_dataset(path: dict, splits: list) -> dict:
    """
    Load the dataset from the path
    Args:
        path: path to the dataset
        splits: list of splits to load

    Returns:
        dataset: dictionary of the dataset
    """
    dataset = {}
    for key, split in zip(path.keys(), splits):
        dataset[key] = load_dataset(path[key], split=f"train[{split[0]}:{split[1]}]")["translation"]
    return dataset


def get_sentences(dataset: dict, languages: list) -> dict:
    """
    Get the sentences from the dataset
    Args:
        dataset: dictionary of the dataset
        languages: list of languages to get the sentences from

    Returns:
        sentences: dictionary of the sentences
    """
    sentences = {}
    for key, lang in zip(dataset.keys(), languages):
        sentences[key] = []
        for pair in dataset[key]:
            sentences[key].append(pair[lang])

    return sentences


def get_avg_len(sentences: dict) -> dict:
    """
    Get the average length of the sentences
    Args:
        sentences: dictionary of the sentences

    Returns:
        avg_len: dictionary of the average length of the sentences
    """
    avg_len = {}
    for key, value in sentences.items():
        avg_len[key] = sum([len(sentence.split()) for sentence in value]) / len(value)
    return avg_len


def tokenize_sentences(sentences: dict, tokenizer: T5TokenizerFast, batch_size=1000) -> dict:
    """
    Tokenize the sentences
    Args:
        sentences: dictionary of the sentences
        tokenizer: tokenizer to use
        batch_size: batch size to use

    Returns:
        tokenized_sentences: dictionary of the tokenized sentences
    """
    tokenized_sentences = {}
    for key, value in sentences.items():
        tokens = []
        for i in range(0, len(value), batch_size):
            tokens.extend(tokenizer(value[i: i + batch_size]).input_ids)
        tokenized_sentences[key] = tokens

    return tokenized_sentences


def get_ngrams(tokenized_sentences: dict) -> tuple[dict, dict, dict]:
    """
    Get the ngrams from the tokenized sentences
    Args:
        tokenized_sentences: dictionary of the tokenized sentences

    Returns:
        unigrams: dictionary of the unigrams
        bigrams: dictionary of the bigrams
        trigrams: dictionary of the trigrams
    """
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


def get_intersection(ngrams: dict) -> dict:
    """
    Get the intersection of the ngrams
    Args:
        ngrams: dictionary of the ngrams

    Returns:
        intersection: dictionary of the intersection of the ngrams
    """
    intersection = {}
    keys = list(ngrams.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            intersection[f"{keys[i]}-{keys[j]}"] = ngrams[keys[i]].intersection(ngrams[keys[j]])
    return intersection


def pretty_print(intersection: dict, ngram: dict, ngram_name: str) -> None:
    """
    Pretty print the intersection of the ngrams
    Args:
        intersection: dictionary of the intersection of the ngrams
        ngram: dictionary of the ngrams
        ngram_name: name of the ngrams
    """
    keys = list(intersection.keys())
    print(f"Analysis of {ngram_name}")
    print('*' * 50)
    # Print the length of the intersections
    for key in keys:
        print(f"{ngram_name} intersection between {key}: {len(intersection[key])}")

    print('-' * 50)
    # Print the length of the difference between the intersections
    for key in keys:
        keys_ = keys.copy()
        keys_.remove(key)
        for key_ in keys_:
            print(f"{ngram_name} difference between {key} and {key_}: "
                  f"{len(intersection[key].difference(intersection[key_]))}")

    print('-' * 50)
    # Print the length of the difference of the union between two intersections and the other
    for key in keys:
        keys_ = keys.copy()
        keys_.remove(key)
        print(f"{ngram_name} the union of  {keys_[0]} and {keys_[1]} diff {key}  : "
              f"{len(intersection[keys_[0]].union(intersection[keys_[1]]).difference(intersection[key]))}")

    print('-' * 50)
    # Print the length of the ngrams
    for key in ngram.keys():
        print(f"{ngram_name} length {key} : {len(ngram[key])}")

    print('*' * 50)


def plot_venn(path: dict, splits: list, languages: list, save: bool, name: str) -> None:
    """
    Plot the venn diagram of the ngrams
    Args:
        path: path to the dataset
        splits: list of splits to load
        languages: list of languages to get the sentences from
        save: whether to save the plot or not
        name: name of the plot
    """
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


def print_stats(path: dict, splits: list, languages: list) -> None:
    """
    Print the stats of the dataset
    Args:
        path: path to the dataset
        splits: list of splits to load
        languages: list of languages to get the sentences from
    """
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
    paths = []
    paths.append({"it": "j0hngou/ccmatrix_en-it", "fr": "j0hngou/ccmatrix_en-fr", "ro": "din0s/ccmatrix_en-ro"})
    paths.append({"it": "j0hngou/ccmatrix_en-it", "de": "j0hngou/ccmatrix_de-en"})
    paths.append({"it": "j0hngou/ccmatrix_en-it", "ithrs": "irenepap/en-it-hrs-data", "itlrs": "irenepap/en-it-lrs-data"})

    splits = []
    splits.append([("", ""), ("", ""), ("", "")])
    splits.append([("", ""), ("", "")])
    splits.append([("", ""), ("12000", ""), ("12000", "")])

    languages = []
    languages.append(["it", "fr", "ro"])
    languages.append(["it", "de"])
    languages.append(["it", "ithrs", "itlrs"])

    names = ["Italian-French-Romanian", "Italian-German", "Italian-HRS-LRS"]

    for path, split, language, name in zip(paths, splits, languages, names):
        # print_stats(path, split, language)
        plot_venn(path, split, language, save=True, name=name)
