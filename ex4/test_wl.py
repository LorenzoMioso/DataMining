import sys
from multiprocessing import Pool

import numpy as np

sys.path.append("..")
from ex4.tree import Best_tree, predict
from ex4.util import (
    class_value,
    compute_tf_idf_row,
    constant_value,
    count_labels,
    equals_next,
    equals_previous,
    frequency_ratio,
    index_tuple,
    parse_dataset,
    progressive_count_labels,
    unique_label_count,
)

TRAIN_TEST_CYCLES = 1000

DATASET_PATH = "../datasets/activity.txt"  # 35 lines
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
    accuracies = []
    count_accuracy_50 = 0

    input_func = [(dataset_path, parse_dataset_func) for _ in range(TRAIN_TEST_CYCLES)]

    # accuracies = [
    #    create_and_test_best_tree(DATASET_PATH, parse_dataset_func)
    #    for _ in range(TRAIN_TEST_CYCLES)
    # ]

    with Pool(16) as p:
        accuracies = p.starmap(create_and_test_best_tree, input_func)

    for acc in accuracies:
        if acc > 0.5:
            count_accuracy_50 += 1

    print(f"Mean accuracy: {np.mean(accuracies)}")
    print(
        f"Percentage of times with accuracy over 50%: {count_accuracy_50 / TRAIN_TEST_CYCLES}"
    )
    print(f"Standard deviation: {np.std(accuracies)}")


def try_different_parse_dataset(dataset_path):
    print("---------------------------")
    print("count_labels parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, count_labels)
    print("---------------------------")
    print("progressive_count_labels parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, progressive_count_labels)
    print("---------------------------")
    print("constant_value parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, constant_value)
    print("---------------------------")
    print("class_value parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, class_value)
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
    print("---------------------------")
    print("equals_next parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, equals_next)
    print("---------------------------")
    print("equals_previous parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, equals_previous)
    print("---------------------------")
    print("tf_idf parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, compute_tf_idf_row)


if __name__ == "__main__":
    datasets = [
        "../datasets/activity.txt",
        "../datasets/question.txt",
        "../datasets/epitope.txt",
        "../datasets/gene.txt",
        "../datasets/robot.txt",
    ]
    for dataset in datasets:
        print("############################################")
        print(f"Dataset: {dataset}")
        print("############################################")
        try_different_parse_dataset(dataset)
