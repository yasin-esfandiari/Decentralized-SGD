import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Set size globally, default is [6.4, 4.8]
plt.rcParams["figure.figsize"] = (6.4, 3.6)

# Choose between 'small' and 'movielens' dataset
DATASET = "small"
# DATASET = "movielens"

root_result_dir = f"output/{DATASET}"

# Number of tests for each parameter combination
num_repeat = 5

# Limit how many iterations are graphed, so that the graphs are consistent
iter_limit = 2000

# Create graphs
for ALPHA in (0.01, 0.001, 0.0001):
    fig, ax = plt.subplots()
    for WORKERS in (1, 2, 4, 8):
        out_dir = f"{root_result_dir}/{WORKERS}/{ALPHA}/"
        df = None
        # Read loss data from each test run and add them to the dataframe 'df'
        for test_id in range(1, num_repeat + 1):
            temp_df = pd.read_csv(f"{out_dir}/{test_id}_losses.tsv", sep="\t", index_col=0)
            # If df is None, only copy, otherwise merge
            if df is None:
                df = temp_df
            else:
                df = pd.merge(df, temp_df, on="iteration", how="outer")
                df.ffill(inplace=True)

        # Compute the average loss value and create x, y data
        df = df.mean(axis=1)
        x = df.index.to_numpy()
        y = df.to_numpy()

        # Ensure that x and y contain iter_limit elements by either cropping
        # or forward filling
        size = x.shape[0]

        if size > iter_limit:
            x = x[:iter_limit]
            y = y[:iter_limit]
        if size < iter_limit:
            last_value = y[size-1]
            x_t = []
            y_t = []
            for i in range(size, iter_limit):
                x_t.append(i)
                y_t.append(last_value)
            x = np.append(x, x_t)
            y = np.append(y, y_t)

        ax.plot(x, y, label=f"D = {WORKERS}", linewidth=2.5)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    fig.savefig(f"{root_result_dir}/{ALPHA}.svg")
    plt.close()
    
