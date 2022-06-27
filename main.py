import numpy as np
import pandas as pd
import argparse

from src.dsgd import DSGD, plotter
from timeit import default_timer as timer

# Choose between 'small' and 'movielens' dataset
DATASET = "small"
# DATASET = "movielens"

# DSDG parameter setup
ITERATIONS = 2000  # Number of iterations
FACTORS = 4  # Number of factors (size R of matrices W and H)
WORKERS = 1  # Number of workers (distributed nodes)
D = 2  # Number of elements the matrices are split to

ALPHA = 0.002  # Gradient step parameter
BETA = 0.02  # Regularization parameter
PATIENCE = 5  # hold on to early stop


def main(dataset, iterations, factors, workers, d, alpha, beta, patience, out_dir, id):

    if dataset == "movielens":
        # MovieLens dataset
        df = pd.read_csv("data/movielens/u1.base", sep="\t",
                         names=["user_id", "item_id", "rating", "timestamp"])

        m = df["user_id"].max()
        n = df["item_id"].max()

        nz = 0
        V = np.zeros((m, n))
        for row in df.itertuples():
            nz += 1
            V[row[1] - 1, row[2] - 1] = row[3]

        print(m, n, nz)
        exit(0)
        # End of MovieLens dataset loading

    elif dataset == "small":
        # Small matrix
        V = np.genfromtxt(fname="data/small/matrix.tsv", delimiter="\t")
        m, n = V.shape
        # End of small matrix loading

    else:
        print("ERROR: Unknown dataset")
        exit(0)

    # Initialize the matrices W0 and H0 randomly
    W0 = np.random.rand(m, factors)
    H0 = np.random.rand(factors, n)

    start = timer()
    converged, losses = DSGD(V, W0, H0, factors, workers, d, iterations, alpha, beta, patience)
    final_loss = losses[-1]
    total_iterations = len(losses)
    elapsed_seconds = timer() - start

    print(f"Finished in {elapsed_seconds} s")

    plotter(losses, out_dir, id)
    with open(f"{out_dir}/{id}_results.tsv", mode="w") as f:
        f.write(f"converged\t{int(converged)}\n"
                f"final_loss\t{final_loss}\n"
                f"iterations\t{total_iterations}\n"
                f"elapsed_seconds\t{elapsed_seconds}\n")

    with open(f"{out_dir}/{id}_losses.tsv", mode="w") as f:
        f.write("iteration\tloss\n")
        for i, loss in enumerate(losses, 1):
            f.write(f"{i}\t{loss}\n")

    return converged, final_loss, total_iterations, elapsed_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default=DATASET)
    parser.add_argument('--iter', type=int, default=ITERATIONS)
    parser.add_argument('--factors', type=int, default=FACTORS)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('-d', type=int, default=D)
    parser.add_argument('--alpha', type=float, default=ALPHA)
    parser.add_argument('--beta', type=float, default=BETA)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--out_dir', type=str, default="./")
    parser.add_argument('--id', type=int, default=0)

    args = parser.parse_args()

    main(dataset=args.data, iterations=args.iter, factors=args.factors,
         workers=args.workers, d=args.d, alpha=args.alpha, beta=args.beta,
         patience=args.patience, out_dir=args.out_dir, id=args.id)
