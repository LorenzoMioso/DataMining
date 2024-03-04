from collections import Counter

import pandas as pd
from numba import jit


def parse_dataset(path):
    items = []
    classes = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip().split("\t")
            items.append(line[1].split(" "))
            classes.append(-1 if line[0] == "0" else int(line[0]))
    return pd.DataFrame(
        [
            # as v in the tuple is the count of the letter in the sequence
            {"s": [(i, l, Counter(s)[l]) for i, l in enumerate(s)], "y": y}
            for s, y in zip(items, classes)
        ]
    )
