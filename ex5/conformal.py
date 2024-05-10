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

SIGNIFICANCE = 0.05


class MyClassifierAdapter(ClassifierAdapter):
    def __init__(self, model: BoostedSeqTree):
        super().__init__(model, None)

    def fit(self, x, y):
        self.model.fit(x, y, ITERATIONS)

    def predict(self, x):
        return np.array(
            [
                np.array(
                    [self.model.predict_prob(x, 1), self.model.predict_prob(x, -1)]
                )
                for x in x
            ]
        )


def main():
    PHI = []
    df = parse_dataset(DATASET_PATH)
    classes = df["y"].unique()

    # divide training and test sets
    df_train = df.sample(frac=0.8)
    df_test = df.drop(df_train.index)
    X_test = df_test["s"].to_numpy()
    Y_test = df_test["y"].to_numpy()

    r = []

    print("classes:", classes)

    for c in classes:
        df_train["y"] = df_train["y"].apply(lambda x: 1 if x == c else 0)

        # split the dataset into train and calibration sets
        df_t = df_train.sample(frac=0.8)
        df_c = df_train.drop(df_t.index)

        # make sure that in the calibration set there are both positive and negative examples
        while df_c["y"].nunique() < 2 or df_t["y"].nunique() < 2:
            df_t = df_train.sample(frac=0.8)
            df_c = df_train.drop(df_t.index)

        X = np.array([np.array([x for x in s]) for s in df_t["s"]])
        Y = np.array(df_t["y"])

        # train the model
        print(f"Training model for class {c}")
        bst = BoostedSeqTree()
        model = MyClassifierAdapter(bst)
        nc = ClassifierNc(model)
        icp = IcpClassifier(nc)

        X_c = np.array([np.array([x for x in s]) for s in df_c["s"]])
        Y_c = np.array(df_c["y"])

        print("call fit")
        icp.fit(X_c, Y_c)

        print(f"Calibrating model for class {c}")
        icp.calibrate(X_c, Y_c)
        PHI.append(icp)

        print("Predicting region")
        test_data = np.array([np.array([x for x in s]) for s in X_test])
        regions = icp.predict(test_data, significance=SIGNIFICANCE)

        print("Regions: ", regions)

        r.append(regions)


if __name__ == "__main__":
    main()
