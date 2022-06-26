import os

from main import main

# Choose between 'small' and 'movielens' dataset
DATASET = "small"
# DATASET = "movielens"

# Create the result directory
root_result_dir = f"output/{DATASET}/"
os.makedirs(root_result_dir, exist_ok=True)

# Number of tests for each parameter combination
num_repeat = 5

# Static parameters
ITERATIONS = 5000  # Number of iterations
FACTORS = 2  # Number of factors (size R of matrices W and H)
BETA = 0.02  # Regularization parameter
PATIENCE = 5  # hold on to early stop

# Run the tests
for WORKERS in (1, 2, 4, 8):
    for ALPHA in (0.1, 0.01, 0.001):
        # Create result folder
        out_dir = f"{root_result_dir}/{WORKERS}/{ALPHA}/"
        os.makedirs(out_dir, exist_ok=True)

        # Write parameters
        with open(f"{out_dir}/parameters.tsv", mode="w") as f:
            f.write(f"dataset\t{DATASET}\n"
                    f"iterations\t{ITERATIONS}\n"
                    f"factors\t{FACTORS}\n"
                    f"workers\t{WORKERS}\n"
                    f"d\t{WORKERS}\n"
                    f"alpha\t{ALPHA}\n"
                    f"beta\t{BETA}\n"
                    f"patience\t{PATIENCE}\n")

        # Run multiple tests
        num_converged = 0
        avg_loss = 0
        avg_iterations = 0
        avg_time = 0

        for test_id in range(1, num_repeat+1):
            converged, final_loss, total_iterations, time = main(dataset=DATASET, iterations=ITERATIONS, factors=FACTORS,
                                                                 workers=WORKERS, d=WORKERS, alpha=ALPHA, beta=BETA,
                                                                 patience=PATIENCE, out_dir=out_dir, id=test_id)
            num_converged += int(converged)
            avg_loss += final_loss
            avg_iterations += total_iterations
            avg_time += time

        # Collect plots and results
        avg_loss /= num_repeat
        avg_iterations /= num_repeat
        avg_time /= num_repeat

        with open(f"{out_dir}/results.tsv", mode="w") as f:
            f.write(f"converged_count\t{num_converged}\n"
                    f"avg_loss\t{avg_loss}\n"
                    f"avg_iterations\t{avg_iterations}\n"
                    f"avg_time_seconds\t{avg_time}\n")
