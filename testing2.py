import itertools

import numpy as np
import torch
import torch.nn as nn


def main():
    list1 = [1, 2, 3, 4, 5]
    list2 = [6, 7, 8, 9]
    for idx, (i, j) in enumerate(itertools.product(list1, list2)):
        print(f"{idx = } | {i = } | {j = } | ")


if __name__ == "__main__":
    main()
