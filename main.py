import numpy as np
import pandas as pd

from src.dsgd import DSGD


# DSDG parameter setup
ITERATIONS = 1000  # Number of iterations
FACTORS = 4  # Number of factors (size R of matrices W and H)
WORKERS = 1  # Number of workers (distributed nodes)
D = 2  # Number of elements the matrices are split to

ALPHA = 0.002  # Gradient step parameter
BETA = 0.02  # Regularization parameter


def main():

    # MovieLens dataset
    # df = pd.read_csv("data/movielens/u1.base", sep="\t",
    #                  names=["user_id", "item_id", "rating", "timestamp"])
    #
    # m = df["user_id"].max()
    # n = df["item_id"].max()
    #
    # V = np.zeros((m, n))
    # for row in df.itertuples():
    #     V[row[1] - 1, row[2] - 1] = row[3]
    # End of MovieLens dataset loading

    # Random matrix
    m = 10
    n = 10
    V = np.random.rand(m, n)
    # End of random matrix loading

    # Initialize the matrices W0 and H0 randomly
    W0 = np.random.rand(m, FACTORS)
    H0 = np.random.rand(FACTORS, n)

    DSGD(V, W0, H0, FACTORS, WORKERS, D, ITERATIONS, ALPHA, BETA)


if __name__ == "__main__":
    main()
