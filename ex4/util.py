import math
import os
import pickle
from collections import Counter

import pandas as pd


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
    max_items=10000,  # per class
    gen_tuple=equals_next,
):

    parsed_df = load_parsed_dataset(path, max_items, gen_tuple)
    if parsed_df is not None:
        return parsed_df

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

    res = None

    match gen_tuple.__name__:
        case compute_tf_idf_row.__name__:
            res = pd.DataFrame(
                [
                    {"s": compute_tf_idf_row(s, labels_count, document_count), "y": y}
                    for s, y in zip(items, classes)
                ]
            )
        case class_value.__name__:
            res = pd.DataFrame(
                [{"s": class_value(s, y), "y": y} for s, y in zip(items, classes)]
            )

        case _:
            res = pd.DataFrame(
                [{"s": gen_tuple(s), "y": y} for s, y in zip(items, classes)]
            )

    save_parsed_dataset(res, path, max_items, gen_tuple)

    return res


# data = parse_dataset("../datasets/activity.txt", 10, count_labels)
# data = parse_dataset("../datasets/activity.txt", 10, progressive_count_labels)
# data = parse_dataset("../datasets/activity.txt", 10, constant_value)
# data = parse_dataset("../datasets/activity.txt", 10, class_value)
# data = parse_dataset("../datasets/activity.txt", 10, index_tuple)
# data = parse_dataset("../datasets/activity.txt", 10, frequency_ratio)
# data = parse_dataset("../datasets/activity.txt", 10, unique_label_count)
# data = parse_dataset("../datasets/activity.txt", 10, equals_next)
# data = parse_dataset("../datasets/activity.txt", 10, equals_previous)
# data = parse_dataset("../datasets/activity.txt", 10, compute_tf_idf_row)
# data = parse_dataset("../datasets/gene.txt", 100000, compute_tf_idf_row)
# data = parse_dataset("../datasets/robot.txt", 100000, compute_tf_idf_row)
# data = parse_dataset("../datasets/epitope.txt", 100000, compute_tf_idf_row)
# for i in range(len(data)):
#    print(data["s"][i])
#    print(data["y"][i])
#    print("--------------------")
