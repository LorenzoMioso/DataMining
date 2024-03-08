from multiprocessing import Pool

import numpy as np
from tree import Best_tree, predict
from util import *

# DATASET_PATH = "../datasets/activity.txt"  # 35 lines
# DATASET_PATH = "../datasets/question.txt"  # 1730 lines
# DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
# DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
# DATASET_PATH = "../datasets/robot.txt"  # 4302 lines


def create_and_test_best_tree(dataset_path, parse_dataset_func=count_labels):
    df = parse_dataset(dataset_path, gen_tuple=parse_dataset_func)
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    W = np.ones(len(train)) / len(train)
    VT = np.zeros(len(train))
    X = train["s"].tolist()
    Y = train["y"].tolist()

    tree = Best_tree(W, VT, X, Y)

    correct = 0
    for _, row in test.iterrows():
        x = row["s"]
        y = row["y"]
        p = predict(tree, (0, x))
        if p == y:
            correct += 1

    return correct / len(test)


def is_valid_weak_learner(dataset_path, parse_dataset_func=count_labels):
    cycles = 100
    accuracies = []
    count_accurancy_50 = 0

    input_func = [(dataset_path, parse_dataset_func) for _ in range(cycles)]

    # accuracies = [
    #    create_and_test_best_tree(DATASET_PATH, parse_dataset_func)
    #    for _ in range(cycles)
    # ]

    with Pool(16) as p:
        accuracies = p.starmap(create_and_test_best_tree, input_func)

    for acc in accuracies:
        if acc > 0.5:
            count_accurancy_50 += 1

    print(f"Mean accuracy: {np.mean(accuracies)}")
    print(f"Percentage of times with accuracy over 50%: {count_accurancy_50 / cycles}")
    print(f"Standard deviation: {np.std(accuracies)}")


def try_different_parse_dataset(dataset_path):
    print("---------------------------")
    print("count_labels parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, count_labels)
    print("---------------------------")
    print("constant_value parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, constant_value)
    print("---------------------------")
    print("index_tuple parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, index_tuple)
    print("---------------------------")
    print("frequency_ratio parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, frequency_ratio)
    print("---------------------------")
    print("unique_label_count parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, unique_label_count)


if __name__ == "__main__":
    datasets = [
        "../datasets/activity.txt",
        "../datasets/question.txt",
        "../datasets/epitope.txt",
        "../datasets/gene.txt",
        "../datasets/robot.txt",
    ]
    for dataset in datasets:
        print(f"############################################")
        print(f"Dataset: {dataset}")
        print(f"############################################")
        try_different_parse_dataset(dataset)
    # main()
