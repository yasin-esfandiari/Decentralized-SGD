import math
import numpy as np
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt

# SGD update paremeters - need to be global so that SGD_update
# can be run in paralel
global ALPHA  # Gradient step parameter
global BETA  # Regularization parameter


def SGD_update(arg):
    global ALPHA, BETA

    # Unpack the argument
    id, V_block, W_block, H_block = arg

    # For each non-zero value in the current block of matrix V, perform
    # SGD update
    for i in range(V_block.shape[0]):
        for j in range(V_block.shape[1]):
            rating = V_block[i][j]
            # Missing data
            if rating == 0:
                continue

            # Select row from W and column from H that are multiplied together
            # to get value at V[i][j]
            Wi = W_block[i, :]
            Hj = H_block[:, j]

            # A squared error loss is used together with regularization to
            # avoid matrices W and H having too large values
            # http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
            eij = rating - np.dot(Wi, Hj)
            W_block[i, :] = Wi + ALPHA * (2 * eij * Hj.T - BETA * Wi)
            H_block[:, j] = Hj + ALPHA * (2 * eij * Wi.T - BETA * Hj)

    return id, W_block, H_block


# L2 loss (nonzero element only)
def L2_loss(V, W, H):
    # Recontruct the matrix V
    V_temp = W.dot(H)
    # Select only the non-zero elements of V (0 corresponds to unknown values)
    nz_index = V.nonzero()
    # Calculate the sum of squares of differences between original and
    # reconstructed values
    difference = np.asarray(V[nz_index] - V_temp[nz_index])
    sum = np.sum(difference ** 2)

    return sum

def plotter(losses):
    epochs = np.arange(1, len(losses)+1)
    plt.plot(epochs, losses, label='Training Loss')

    # plt.xticks(epochs)
    # plt.yticks(np.arange(np.max(losses)+5, np.min(losses)-5, 500))
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()

    plt.show()
    plt.savefig("plot.png")


def DSGD(V, W0, H0, factors, workers=1, d=1, max_iterations=100, alpha=0.002, beta=0.02, patience=5):
    # Set SGD update parameters
    global ALPHA, BETA
    ALPHA = alpha
    BETA = beta
    counter = 0 # Early stopping counter
    best_loss = sys.maxsize # initialization for the max possible int value for loss
    losses = [] # keep track of the losses

    m, n = V.shape

    # b_m, b_n are number of rows and columns in each block of the matrix V
    b_m = math.ceil(m / d)
    b_n = math.ceil(n / d)

    # Create list of block range indices
    # I_m and I_n contain slices representing indices of start and end of
    # each block along the m, n dimensions respectively
    # Example: for m = 5, n = 7, d = 3 the slices are
    # I_m = [(0:2), (2:4), (4:5)]
    # I_n = [(0:3), (3:6), (6:7)]
    I_m = [slice(i * b_m, (i + 1) * b_m) for i in range(d - 1)]
    I_n = [slice(i * b_n, (i + 1) * b_n) for i in range(d - 1)]
    # Add the last slice
    I_m.append(slice((d - 1) * b_m, m))
    I_n.append(slice((d - 1) * b_n, n))

    W = W0.copy()
    H = H0.copy()

    W_new = W0.copy()
    H_new = H0.copy()

    with mp.Pool(workers) as pool:
        # Run untill number of itertaions provided by user is reached
        iterations = 0
        while iterations < max_iterations:

            # Subepoch
            for k in range(d):

                # Get the indices of substrata forming the current stratum
                # The index calculation for 'cols' will create an alternating
                # pattern, that ensures the substrata are interchangeable
                # After 'workers' steps, the strata will cover the whole
                # training set
                # Example:
                #   k=0           k=1           k=2
                # |x| | |       | |x| |       | | |x|
                # | |x| |       | | |x|       |x| | |
                # | | |x|       |x| | |       | |x| |
                rows = [i for i in range(d)]
                cols = [(k + i) % d for i in range(d)]

                # Select blocks of matrix Vib and parameters Wib and Hjb
                W_b = [W[I_m[row], :] for row in rows]
                H_b = [H[:, I_n[col]] for col in cols]
                V_b = [V[I_m[row], I_n[col]] for (row, col) in zip(rows, cols)]

                # Prepare data for paralel execution in the form
                # (id, block V, block W, block H)
                # id is needed to know which block is returned by which process
                data = []
                for b in range(d):
                    data.append([b, V_b[b], W_b[b], H_b[b]])

                # Calculate the SGD update
                result = pool.map(SGD_update, data)
                for r in result:
                    id, W_, H_ = r
                    W_new[I_m[rows[id]], :] = W_
                    H_new[:, I_n[cols[id]]] = H_

            iterations += 1

            W = W_new.copy()
            H = H_new.copy()

            # For now, print the L2 loss to terminal only
            loss = L2_loss(V, W_new, H_new)
            print(f"Iteration: {iterations}/{max_iterations}\tLoss: {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                counter = 0
                losses.append(loss)
            else:
                counter += 1
                print('loss at patience level ', counter, ' is ', loss)
                if (counter == patience):
                    print("Early stopping with best loss : ", best_loss, f" for the iteration ({iterations-counter}): ")
                    break
            
    return losses

