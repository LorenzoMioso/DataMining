import math
from collections import Counter

import pandas as pd


# as v in the tuple is the count of the labels in the sequence
def count_labels(s):
    return [(i, l, Counter(s)[l]) for i, l in enumerate(s)]


# as v in the tuple is the constant value 0
def constant_value(s, c=0):
    return [(i, l, c) for i, l in enumerate(s)]


# as v in the tuple is the index of the label in the sequence
def index_tuple(s):
    return [(i, l, i) for i, l in enumerate(s)]


# as v in the tuple is the frequency of the label in the sequence
def frequency_ratio(s):
    total_count = len(s)
    return [(i, l, count / total_count) for i, l, count in count_labels(s)]


# as v in the tuple is the frequency of the label in the sequence
def unique_label_count(s):
    unique_labels = set(s)
    return [(i, l, len(unique_labels)) for i, l in enumerate(s)]


# as v in the tuple is the frequency of the label in the sequence
def equals_next(s):
    return [
        (i, l, 1 if i < len(s) - 1 and s[i] == s[i + 1] else 0) for i, l in enumerate(s)
    ]


# as v in the tuple is the frequency of the label in the sequence
def equals_previous(s):
    return [(i, l, 1 if i > 0 and s[i] == s[i - 1] else 0) for i, l in enumerate(s)]


# as v in the tuple is the tf-idf of the label in the sequence (document)
def compute_tf_idf_row(s, labels_count, document_count):
    return [
        (i, l, (count / len(s)) * math.log(document_count / labels_count[l]))
        for i, l, count in count_labels(s)
    ]


def parse_dataset(path, max_items=100000, gen_tuple=frequency_ratio):
    items = []
    classes = []
    count_items_per_class = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().split("\t")
            l = line[1].split(" ")
            c = -1 if line[0] == "0" else int(line[0])

            if c not in count_items_per_class:
                count_items_per_class[c] = 0
            if count_items_per_class[c] < max_items:
                count_items_per_class[c] += 1
                items.append(l)
                classes.append(c)

    return pd.DataFrame([{"s": gen_tuple(s), "y": y} for s, y in zip(items, classes)])


def parse_dataset_tf_idf(path):
    # the value in each tuple is the tf-idf of the label in the sequence (document)

    items = []
    classes = []
    labels_count = {}
    document_count = 0

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().split("\t")
            l = line[1].split(" ")
            c = -1 if line[0] == "0" else int(line[0])

            document_count += 1
            for label in l:
                if label not in labels_count:
                    labels_count[label] = 0
                labels_count[label] += 1

            items.append(l)
            classes.append(c)

    return pd.DataFrame(
        [
            {"s": compute_tf_idf_row(s, labels_count, document_count), "y": y}
            for s, y in zip(items, classes)
        ]
    )


# TODO : croos correlation
