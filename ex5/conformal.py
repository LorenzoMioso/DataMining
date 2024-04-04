import sys

sys.path.append("..")

import numpy as np

from ex4 import ada_boost as ab
from ex5.util import parse_dataset

DATASET_PATH = "../datasets/pioneer.txt"  ## 160 lines
# DATASET_PATH = "../datasets/auslan2.txt"  ## 200 lines
# DATASET_PATH = "../datasets/context.txt"  ## 240 lines
# DATASET_PATH = "../datasets/aslbu.txt"  #### 424 lines
# DATASET_PATH = "../datasets/skating.txt"  ## 530 lines
# DATASET_PATH = "../datasets/reuters.txt"  # 1010 lines
# DATASET_PATH = "../datasets/webkb.txt"  ### 3667 lines
# DATASET_PATH = "../datasets/news.txt"  #### 4976 lines
# DATASET_PATH = "../datasets/unix.txt"  #### 5472 lines
ITERATIONS = 50


def build_conformal_classifier(X, Y):
    pass


def main():
    df = parse_dataset(DATASET_PATH)

    classes = df["y"].unique()

    for c in classes:
        print(f"Class: {c}")
        df["y"].apply(lambda x: 1 if x == c else 0, inplace=True)

        print(df.head())

        # split the dataset into train and calibration sets
        train = df.sample(frac=0.8)
        calibrate = df.drop(train.index)

        X = train["s"].tolist()
        Y = train["y"].tolist()

        # train the model
        t, a, _ = ab.ada_boost(X, Y, ITERATIONS)

        # calibrate the model


if __name__ == "__main__":
    main()
