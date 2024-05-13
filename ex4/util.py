import math
import os
import pickle

# ignore the warning from numpy
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# define the datatype of a sequence element
SEQ_ITEM_TYPE = np.dtype([("d", "int32"), ("l", "U50"), ("v", "float32")])


# as v in the tuple is the total count of the labels in the sequence
def count_labels(s):
    # Captures some global information about the sequence,
    # idipendently of the position of the label in the sequence
    return [(i, l, Counter(s)[l]) for i, l in enumerate(s)]


# as v in the tuple is the count of the labels in the sequence
def progressive_count_labels(s):
    # Captures some local information about the sequence,
    # Saves some information from the previous labels
    return [(i, l, sum(1 for j in range(i + 1) if s[j] == l)) for i, l in enumerate(s)]


# as v in the tuple is the constant value 0
def constant_value(s, c=0):
    # Does not capture any information about the sequence
    return [(i, l, c) for i, l in enumerate(s)]


def class_value(s, c):
    # Tells the class of the sequence
    return [(i, l, c) for i, l in enumerate(s)]


# as v in the tuple is the index of the label in the sequence
def index_tuple(s):
    # Tells the position of the label in the sequence
    return [(i, l, i) for i, l in enumerate(s)]


# as v in the tuple is the frequency of the label in the sequence
def frequency_ratio(s):
    # The same as count_labels
    total_count = len(s)
    return [(i, l, count / total_count) for i, l, count in count_labels(s)]


# as v in the tuple is the frequency of the label in the sequence
def unique_label_count(s):
    # Some global information about the sequence
    unique_labels = set(s)
    return [(i, l, len(unique_labels)) for i, l in enumerate(s)]


# as v in the tuple is the frequency of the label in the sequence
def equals_next(s):
    # A local information about the sequence
    return [
        (i, l, 1 if i < len(s) - 1 and s[i] == s[i + 1] else 0) for i, l in enumerate(s)
    ]


# as v in the tuple is the frequency of the label in the sequence
def equals_previous(s):
    # A local information about the sequence
    return [(i, l, 1 if i > 0 and s[i] == s[i - 1] else 0) for i, l in enumerate(s)]


# as v in the tuple is the tf-idf of the label in the sequence (document)
def compute_tf_idf_row(s, labels_count, document_count):
    # A global information about the sequence and the dataset
    return [
        (i, l, (count / len(s)) * math.log(document_count / labels_count[l]))
        for i, l, count in count_labels(s)
    ]


def load_parsed_dataset(path, max_items=100, gen_tuple=frequency_ratio):
    # check if a file with the parsed dataset already exists
    try:
        dataset_path = path.replace(".txt", "")
        with open(f"{dataset_path}/{max_items}_{gen_tuple.__name__}.pkl", "rb") as file:
            return pickle.load(file)
    except:
        return None


def save_parsed_dataset(data, path, max_items=100, gen_tuple=frequency_ratio):
    # create directory if it does not exist

    dataset_path = path.replace(".txt", "")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # save as pickle
    with open(f"{dataset_path}/{max_items}_{gen_tuple.__name__}.pkl", "wb") as file:
        pickle.dump(data, file)


def parse_dataset(
    path,
    max_items=100000,  # per class
    gen_tuple=count_labels,
    add_padding=True,
):

    # parsed_df = load_parsed_dataset(path, max_items, gen_tuple)
    # if parsed_df is not None:
    #    return parsed_df

    items = []
    classes = []
    count_items_per_class = {}
    labels_count = {}
    document_count = 0

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().split("\t")
            l = line[1].split(" ")
            c = -1 if line[0] == "0" else int(line[0])

            if count_items_per_class.get(c, 0) < max_items:
                count_items_per_class[c] = count_items_per_class.get(c, 0) + 1
                items.append(l)
                classes.append(c)

                document_count += 1
                for label in l:
                    if label not in labels_count:
                        labels_count[label] = 0
                    labels_count[label] += 1

    # if add_padding:
    #    # add padding to the sequences
    #    max_len = max([len(s) for s in items])
    #    labels_count["<PAD>"] = 0
    #    for idx, item in enumerate(items):
    #        items[idx] = item + ["<PAD>"] * (max_len - len(item))
    #        labels_count["<PAD>"] += max_len - len(item)

    # res = None

    match gen_tuple.__name__:
        case compute_tf_idf_row.__name__:
            res = np.array(
                [
                    [
                        np.array(
                            gen_tuple(s, labels_count, document_count),
                            dtype=SEQ_ITEM_TYPE,
                        ),
                        y,
                    ]
                    for s, y in zip(items, classes)
                ],
                dtype=object,
            )
        case class_value.__name__:
            res = np.array(
                [
                    [np.array(class_value(s, y), dtype=SEQ_ITEM_TYPE), y]
                    for s, y in zip(items, classes)
                ],
                dtype=object,
            )

        case _:
            res = np.array(
                [
                    [np.array(gen_tuple(s), dtype=SEQ_ITEM_TYPE), y]
                    for s, y in zip(items, classes)
                ],
                dtype=object,
            )

    if add_padding:
        # get max length of the sequences
        max_len = 0
        for s, _ in res:
            max_len = max(max_len, len(s))
        # pad the each sequence to reach the max length with (-1, "", 0)
        res = np.array(
            [
                [
                    np.array(
                        s.tolist() + [(i, "", 0) for i in range(len(s), max_len)],
                        dtype=SEQ_ITEM_TYPE,
                    ),
                    y,
                ]
                for s, y in res
            ],
            dtype=object,
        )

    # save_parsed_dataset(res, path, max_items, gen_tuple)

    return res
