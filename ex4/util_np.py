from collections import Counter

import numpy as np
import pandas as pd


def parse_row(row):
    print(row)
    cls = np.int32(-1 if row[0] == "0" else row[0])
    seq = np.array(
        [
            (np.int32(i), l, np.int32(Counter(row[1])[l]))
            for i, l in enumerate(row[1].split(" "))
        ],
        dtype=[("d", np.int32), ("l", "U1"), ("v", np.int32)],
    )
    print(seq)
    print(cls)
    return {"s": seq, "y": cls}


def parse_dataset(path):

    data = np.genfromtxt(path, delimiter="\t", dtype=str)
    np.apply_along_axis(parse_row, 1, data)

    return data


data = parse_dataset("../datasets/activity.txt")
print(data[0]["s"])
