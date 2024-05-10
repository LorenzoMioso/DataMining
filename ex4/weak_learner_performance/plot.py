import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data = pd.read_csv("weak_learner_perf.csv")
# data = pd.read_csv("10_full.csv")
# data = pd.read_csv("10_full_pad_before.csv")
data = pd.read_csv("10_full_pad_after.csv")


for dataset in data["dataset"].unique():
    subset = data[data["dataset"] == dataset]
    x = subset["v_function"].unique()
    y1 = []
    y2 = []
    for v_function in x:
        sub_subset = subset[subset["v_function"] == v_function]
        y1.append(round(sub_subset["mean_accuracy"].values[0], 3))
        y2.append(
            round(sub_subset["percentage_of_time_with_accuracy_over_50"].values[0], 3)
        )

    # sort by mean accuracy
    y1, y2, x = zip(*sorted(zip(y1, y2, x)))

    x_arr = np.arange(len(x))
    width = 0.3
    multiplier = 0.5

    fig, ax = plt.subplots(layout="constrained")

    rect1 = ax.bar(x_arr + width * (multiplier), y1, width, label="Mean accuracy")
    ax.bar_label(rect1, padding=3)
    multiplier += 1

    rect2 = ax.bar(
        x_arr + width * (multiplier),
        y2,
        width,
        label="Percentage of times with accuracy over 50%",
    )
    ax.bar_label(rect2, padding=3)

    ax.set_xticks(x_arr + width)
    ax.set_xticklabels(x)
    ax.legend()

    plt.title(f"Dataset: {dataset}, (sorted by accuracy)")
    plt.show()
