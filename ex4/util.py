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


def parse_dataset(path, max_items=100000, gen_tuple=frequency_ratio):
    items = []
    classes = []
    count_items_per_class = {}
    with open(path, "r") as file:
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
