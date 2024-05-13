import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

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


data = parse_dataset("../datasets/skating.txt", gen_tuple=count_labels)
