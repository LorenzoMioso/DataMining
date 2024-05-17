import sys
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import train_test_split

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


def create_and_test_best_tree(
    dataset_path, parse_dataset_func=count_labels, add_padding=False
):
    dataset = parse_dataset(
        dataset_path, gen_tuple=parse_dataset_func, add_padding=add_padding
    )
    X = dataset[:, 0]
    Y = dataset[:, 1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    W = np.ones(len(X_train)) / len(X_train)
    VT = np.zeros(len(X_train), dtype=int)
    tree = Best_tree(W, VT, X_train, Y_train)

    correct = 0
    for x, y in zip(X_test, Y_test):
        p = predict(tree, (0, x))
        if p == y:
            correct += 1

    return correct / len(X_test)


def is_valid_weak_learner(
    dataset_path, parse_dataset_func=count_labels, add_padding=False
):
    accuracies = []
    count_accuracy_50 = 0

    input_func = [
        (dataset_path, parse_dataset_func, add_padding)
        for _ in range(TRAIN_TEST_CYCLES)
    ]

    # accuracies = [
    #    create_and_test_best_tree(dataset_path, parse_dataset_func, add_padding)
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
    print(f"Standard deviation: {np.std(accuracies)}", flush=True)


def try_different_parse_dataset(dataset_path, add_padding=False):
    print("---------------------------")
    print("count_labels parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, count_labels, add_padding)
    print("---------------------------")
    print("progressive_count_labels parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, progressive_count_labels, add_padding)
    print("---------------------------")
    print("constant_value parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, constant_value, add_padding)
    print("---------------------------")
    print("class_value parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, class_value, add_padding)
    print("---------------------------")
    print("index_tuple parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, index_tuple, add_padding)
    print("---------------------------")
    print("frequency_ratio parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, frequency_ratio, add_padding)
    print("---------------------------")
    print("unique_label_count parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, unique_label_count, add_padding)
    print("---------------------------")
    print("equals_next parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, equals_next, add_padding)
    print("---------------------------")
    print("equals_previous parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, equals_previous, add_padding)
    print("---------------------------")
    print("tf_idf parse_dataset")
    print("---------------------------")
    is_valid_weak_learner(dataset_path, compute_tf_idf_row, add_padding)


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

    print("############################################")
    print("############# With padding #################")
    print("############################################")
    for dataset in datasets:
        print("############################################")
        print(f"Dataset: {dataset}")
        print("############################################")
        try_different_parse_dataset(dataset, True)
