import os

from generate_parity_dataset import gen_parity_data
from train_mlp import train_mlp

# Grid Search over NN training hyperparameters
#   call train_mlp.py
# For each trained model, extract rule set
# save all rule sets
# create graph for rule sets: dimensions are complexity and accuracy


def main():

    k = 5
    # Generate dataset and save it somewhere
    data_name = "./runs/parity5/data.csv"
    _ = gen_parity_data(data_name, k, True, 1000)
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    n_layers = [1, 2, 3]
    train_mlp(
        data_name,
    )


if __name__ == "__main__":
    main()
