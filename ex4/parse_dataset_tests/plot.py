import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("weak_learner_perf.csv")


for dataset in data["dataset"].unique():
    subset = data[data["dataset"] == dataset]
    x = subset["v_function"].unique()
    y1 = []
    y2 = []
    for v_function in x:
        sub_subset = subset[subset["v_function"] == v_function]
        y1.append(sub_subset["mean_accuracy"].values[0])
        y2.append(sub_subset["percentage_of_time_with_accuracy_over_50"].values[0])

    x_arr = np.arange(len(x))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout="unconstrained")
