import math
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.append("..")

from ex4.util import (
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
    add_padding=False,
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

    if add_padding:
        max_len = max([len(s) for s in items])
        res["s"] = res["s"].apply(lambda x: x + [(-1, "", 0)] * (max_len - len(x)))

    # save_parsed_dataset(res, path, max_items, gen_tuple)

    return res



