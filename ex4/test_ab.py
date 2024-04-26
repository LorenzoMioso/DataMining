import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

sys.path.append("..")
from ex4.ada_boost import BoostedSeqTree
from ex4.util import parse_dataset

ITERATIONS = 40

# DATASET_PATH = "../datasets/activity.txt"  # 35 lines
# DATASET_PATH = "../datasets/question.txt"  # 1730 lines
# DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
# DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
DATASET_PATH = "../datasets/robot.txt"  # 4302 lines


def plot_iterations_accuracy(df, iterations, dataset_path, has_padding):
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    X = train["s"].tolist()
    Y = train["y"].tolist()

    # train a boosting classifier with a given number of iterations
    model = BoostedSeqTree()
    model.fit(X, Y, iterations)

    accuracy = []
    for i in range(1, len(model.trees) + 1):
        test_X = test["s"].tolist()
        test_Y = test["y"].tolist()
        # print(f"iteration = {i}")
        # print(f"text_y = {test_Y}")
        predictions = [
            BoostedSeqTree(model.trees[:i], model.a[:i], model.e[:i]).predict(x)
            for x, y in zip(test_X, test_Y)
        ]
        # print(f"predictions = {predictions}")
        accuracy.append(
            sum([1 for i in range(len(test_Y)) if predictions[i] * test_Y[i] > 0])
            / len(test_Y)
        )
        print(f"Accuracy = {accuracy[-1]}")

    print(f"accuracy = {accuracy}")
    plt.plot(range(1, len(model.trees) + 1), accuracy, label="accuracy")
    plt.plot(range(1, len(model.trees) + 1), model.a, label="alpha")
    plt.plot(range(1, len(model.trees) + 1), model.e, label="error")
    # legend
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title(
        f"AdaBoost on {dataset_path.split('/')[-1].split('.')[0]}, has padding {has_padding}"
    )

    return plt


def run_ada_boost(dataset_path, add_padding=True):
    df = parse_dataset(dataset_path, add_padding=add_padding)
    plt = plot_iterations_accuracy(df, ITERATIONS, dataset_path, add_padding)

    # create directory if it does not exist

    if not os.path.exists(
        f"ada_boost_performance/{dataset_path.split('/')[-1].split('.')[0]}/has_padding_{add_padding}"
    ):
        os.makedirs(
            f"ada_boost_performance/{dataset_path.split('/')[-1].split('.')[0]}/has_padding_{add_padding}"
        )

    # save plot
    print("saving plot")
    plt.savefig(
        f"ada_boost_performance/{dataset_path.split('/')[-1].split('.')[0]}/has_padding_{add_padding}/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"
    )

    plt.close()


if __name__ == "__main__":
    datasets = [
        "../datasets/activity.txt",
        "../datasets/question.txt",
        "../datasets/epitope.txt",
        "../datasets/gene.txt",
        "../datasets/robot.txt",
    ]
    for dataset in datasets:
        run_ada_boost(dataset, add_padding=True)
        run_ada_boost(dataset, add_padding=False)
