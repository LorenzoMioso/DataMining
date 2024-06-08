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
        self.model.fit(x.tolist(), y.tolist(), ITERATIONS)

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
    # consider each class separately
    classes = np.unique(Y)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for c in classes:
        data = X[Y == c]
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
            X_t, X_c, Y_t, Y_c = train_test_split(X_cl, Y_cl, test_size=0.2)

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

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.PHI, self.bsts), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.PHI, self.bsts = pickle.load(f)


def main():
    df = parse_dataset(DATASET_PATH, add_padding=True)
    # divide training and test sets
    X, X_test, Y, Y_test = custom_train_test_split(
        df["s"].to_numpy(), df["y"].to_numpy()
    )

    model = ConformalBoostedSeqTree()
    model.fit(X, Y)
    model.save()

    SIGNIFICANCE = 0.2

    predictions = model.predict(X_test, SIGNIFICANCE)
    for p, y in zip(predictions, Y_test):
        print(f"real value = {y}, prediction = {p}")


if __name__ == "__main__":
    main()
