import sys

sys.path.append("..")

import numpy as np
from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc, NcFactory

from ex4.ada_boost import BoostedSeqTree
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


class MyClassifierAdapter(ClassifierAdapter):
    def __init__(self, model: BoostedSeqTree, fit_params=None):
        super(MyClassifierAdapter, self).__init__(model, fit_params)

    def fit(self, X, Y):
        self.model.fit(X, Y, ITERATIONS)

    def predict(self, X):
        return self.model.predict_prob(X, 1)


def main():
    df = parse_dataset(DATASET_PATH)
    # divide training and test sets
    df_train = df.sample(frac=0.8)
    df_test = df.drop(df_train.index)

    classes = df["y"].unique()

    for c in classes:
        print(f"Class: {c}")
        df_train["y"].apply(lambda x: 1 if x == c else 0, inplace=True)

        print(df_train.head())

        # split the dataset into train and calibration sets
        df_t = df_train.sample(frac=0.8)
        df_c = df_train.drop(df_t.index)

        # make sure that in the calibration set there are both positive and negative examples
        while df_c["y"].nunique() < 2:
            print("Resampling calibration set")
            df_t = df_train.sample(frac=0.8)
            df_c = df_train.drop(df_t.index)

        X = df_t["s"].tolist()
        Y = df_t["y"].tolist()

        # train the model
        bst = BoostedSeqTree().fit(X, Y, ITERATIONS)

        # calibrate the model
        model = MyClassifierAdapter(bst)
        nc = ClassifierNc(model)
        icp = IcpClassifier(nc)

        X_c = df_c["s"].tolist()
        Y_c = df_c["y"].tolist()

        icp.fit(X_c, Y_c)


if __name__ == "__main__":
    main()
