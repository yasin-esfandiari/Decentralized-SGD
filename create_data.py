import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Set size globally, default is [6.4, 4.8]
plt.rcParams["figure.figsize"] = (6.4, 3.6)

# Choose between 'small' and 'movielens' dataset
DATASET = "small"
# DATASET = "movielens"

root_result_dir = f"/media/storage/Stažené/{DATASET}"

# Number of tests for each parameter combination
num_repeat = 5


# Create graphs
for ALPHA in (0.01, 0.001, 0.0001):
    for WORKERS in (1, 2, 4, 8):
        out_dir = f"{root_result_dir}/{WORKERS}/{ALPHA}"
        with open(f"{out_dir}/results.tsv", mode="r") as f:
            next(f)
            loss = float(next(f).strip().split()[1])
            iter = float(next(f).strip().split()[1])
            time = float(next(f).strip().split()[1])

            avg_time = 1000*time/iter

            print("{} & {:.3f} & {} & {:.2f}".format(WORKERS, loss, iter, avg_time))
