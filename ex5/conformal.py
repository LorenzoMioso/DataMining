import sys

from ex5.util import parse_dataset

sys.path.append("..")
import datetime

import dill as pickle
import numpy as np
from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc
from sklearn.model_selection import train_test_split

from ex4.ada_boost import BoostedSeqTree


class MyClassifierAdapter(ClassifierAdapter):
    def __init__(self, model: BoostedSeqTree):
        super().__init__(model, None)

    def fit(self, x, y, ITERATIONS=20):
        if isinstance(x, np.ndarray):
            self.model.fit(x.tolist(), y.tolist(), ITERATIONS)
        else:
            self.model.fit(x, y, ITERATIONS)

    def predict(self, x):
        if isinstance(x[0], np.ndarray):
            return np.array(
                [
                    np.array(
                        [
                            self.model.predict_prob(
                                [(int(i), s, float(f)) for i, s, f in seq], 1
                            ),
                            self.model.predict_prob(
                                [(int(i), s, float(f)) for i, s, f in seq], -1
                            ),
                        ]
                    )
                    for seq in x
                ]
            )
        else:
            return np.array(
                [
                    np.array(
                        [
                            self.model.predict_prob(seq, 1),
                            self.model.predict_prob(seq, -1),
                        ]
                    )
                    for seq in x
                ]
            )


def custom_train_test_split(X, Y, ratio=0.8):
    print(f"Splitting dataset with ratio {ratio}")
    print(f"Dataset size: {len(X)}")
    # consider each class separately
    classes = np.unique(Y)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    classes_size = {c: len(X[Y == c]) for c in classes}
    for c in classes:
        print(f"Class {c}")
        data = X[Y == c]
        classes_size[c] = len(data)
        print(f"data size: {len(data)}")
        x_train, x_test = np.split(data, [int(ratio * len(data))])
        print(f"Class {c}: {len(x_train)} train, {len(x_test)} test")
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(np.full(len(x_train), c))
        Y_test.append(np.full(len(x_test), c))

    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)
    return X_train, X_test, Y_train, Y_test


def bagging(X, Y):
    # Y has values 1 and -1
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    # for train pick len(Y == 1) random samples from Y == 1 and len(Y == -1) random samples from Y == -1
    # for test pick the rest
    pos_indices = np.where(Y == 1)[0]
    neg_indices = np.where(Y == -1)[0]
    used_indices = []
    for _ in range(len(pos_indices)):
        idx = np.random.choice(pos_indices)
        X_train.append(X[idx])
        Y_train.append(Y[idx])
        used_indices.append(idx)
    for _ in range(len(neg_indices)):
        idx = np.random.choice(neg_indices)
        X_train.append(X[idx])
        Y_train.append(Y[idx])
        used_indices.append(idx)

    for i in range(len(X)):
        if i not in used_indices:
            X_test.append(X[i])
            Y_test.append(Y[i])

    # retry if Y_train does not contain both 1 and -1
    if len(np.unique(Y_train)) != 2 or len(np.unique(Y_test)) != 2:
        print("Retrying")
        return bagging(X, Y)

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)


class ConformalBoostedSeqTree:

    ITERATIONS = 20

    def __init__(self):
        self.PHI = {}
        self.bsts = {}

    def fit(self, X, Y):
        classes = np.unique(Y)

        for c in classes:
            print("class:", c)
            X_cl = np.copy(X)
            Y_cl = np.array([1 if y == c else -1 for y in Y])

            # split the dataset into train and calibration sets
            # X_t, X_c, Y_t, Y_c = train_test_split(X_cl, Y_cl, test_size=0.2)

            # split the dataset into train and calibration sets
            X_t, X_c, Y_t, Y_c = bagging(X_cl, Y_cl)

            # make sure that in the calibration set there are both positive and negative examples
            if len(np.unique(Y_t)) != 2 or len(np.unique(Y_c)) != 2:
                raise ValueError("Failed to split correctly")

            # train the model
            print(f"Training model for class {c}")
            bst = BoostedSeqTree()
            model = MyClassifierAdapter(bst)
            nc = ClassifierNc(model)
            icp = IcpClassifier(nc)

            print("Fitting Classifier")
            icp.fit(X_t, Y_t)

            print(f"Calibrating model for class {c}")
            icp.calibrate(X_c, Y_c)
            self.bsts[c] = bst
            self.PHI[c] = icp

        return self

    def predict(self, X, SIGNIFICANCE=0.2):
        X = np.array([np.array(x) for x in X])

        predictions = [{} for _ in range(len(X))]
        for c, phi in self.PHI.items():
            regions = phi.predict(X, significance=SIGNIFICANCE).tolist()
            for i, r in enumerate(regions):
                predictions[i][c] = None
                # positive region
                if r[0] and not r[1]:
                    predictions[i][c] = True
                # negative region
                elif not r[0] and r[1]:
                    predictions[i][c] = False
                # unknown region
                elif r[0] and r[1]:
                    predictions[i][c] = True

        # take only true classes and sort them by class
        res = []
        for p in predictions:
            res.append([k for k, v in sorted(p.items()) if v is not None])

        return res

    def predict_alternative(self, X):
        # use bsts to predict the class of the sequence
        predictions = []
        for x in X:
            p = {}
            for c, bst in self.bsts.items():
                # this returns 1 if the sequence belongs to the class c, -1 otherwise
                p[c] = bst.predict(x)
            predictions.append(p)

        # take only true classes and sort them by class
        res = []
        for p in predictions:
            res.append([k for k, v in sorted(p.items()) if v == 1])

        return res

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.PHI, self.bsts), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.PHI, self.bsts = pickle.load(f)


def main():
    # DATASET_PATH = "../datasets/pioneer.txt"  ## 160 lines
    # DATASET_PATH = "../datasets/auslan2.txt"  ## 200 lines
    DATASET_PATH = "../datasets/context.txt"  ## 240 lines
    # DATASET_PATH = "../datasets/aslbu.txt"  #### 424 lines
    # DATASET_PATH = "../datasets/skating.txt"  ## 530 lines
    # DATASET_PATH = "../datasets/reuters.txt"  # 1010 lines
    # DATASET_PATH = "../datasets/webkb.txt"  ### 3667 lines
    # DATASET_PATH = "../datasets/news.txt"  #### 4976 lines
    # DATASET_PATH = "../datasets/unix.txt"  #### 5472 lines

    # other datasets with 2 classes
    # DATASET_PATH = "../datasets/activity.txt"  # 35 lines
    # DATASET_PATH = "../datasets/question.txt"  # 1730 lines
    # DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
    # DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
    # DATASET_PATH = "../datasets/robot.txt"  # 4302 lines

    df = parse_dataset(DATASET_PATH, add_padding=True)
    # divide training and test sets
    X, X_test, Y, Y_test = custom_train_test_split(
        df["s"].to_numpy(), df["y"].to_numpy()
    )

    model = ConformalBoostedSeqTree()
    model.fit(X, Y)
    model.save(f"conformal_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")

    SIGNIFICANCE = 0.2

    predictions = model.predict(X_test, SIGNIFICANCE)
    for p, y in zip(predictions, Y_test):
        print(f"real value = {y}, prediction = {p}")


if __name__ == "__main__":
    main()
