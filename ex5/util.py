import math
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.append("..")

from ex4.util import (
    SEQ_ITEM_TYPE,
    class_value,
    compute_tf_idf_row,
    count_labels,
    load_parsed_dataset,
    save_parsed_dataset,
)


def parse_dataset(
    path,
    max_items=10000,  # per class
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
            c = line[0]

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
                    {
                        "s": np.array(
                            compute_tf_idf_row(s, labels_count, document_count),
                            dtype=SEQ_ITEM_TYPE,
                        ),
                        "y": y,
                    }
                    for s, y in zip(items, classes)
                ]
            )
        case class_value.__name__:
            res = pd.DataFrame(
                [
                    {"s": np.array(class_value(s, y), dtype=SEQ_ITEM_TYPE), "y": y}
                    for s, y in zip(items, classes)
                ]
            )

        case _:
            res = pd.DataFrame(
                [
                    {"s": np.array(gen_tuple(s), dtype=SEQ_ITEM_TYPE), "y": y}
                    for s, y in zip(items, classes)
                ]
            )

    if add_padding:
        # get max length of the sequences
        max_len = 0
        for r in res["s"]:
            if len(r) > max_len:
                max_len = len(r)

        # pad the each sequence to reach the max length with (-1, "", 0)
        for i in range(len(res)):
            s = res["s"][i]
            if len(s) < max_len:
                res["s"][i] = np.concatenate(
                    (
                        s,
                        np.array(
                            [(-1, "", 0)] * (max_len - len(s)), dtype=SEQ_ITEM_TYPE
                        ),
                    )
                )

    # save_parsed_dataset(res, path, max_items, gen_tuple)

    return res


data = parse_dataset("../datasets/skating.txt", gen_tuple=count_labels)
