import torch.nn as nn

from models.lenet import LeNet, TernaryLeNet
from models.adult import AdultNet, TernaryAdultNet
from models.mushroom import MushroomNet, TernaryMushroomNet
from models.logical_AND import ANDNet, TernaryANDNet


def ModelFactory(
    data: str, ternary: bool, a: float = 0, b: float = 0
) -> nn.Module:
    if data == "mnist":
        n_classes = 10
        return LeNet(n_classes) if not ternary else TernaryLeNet(n_classes, a, b)
    elif data == "adult":
        """You get 104 input features if you one hot encode all independent variables
        AFTER removing the rows with ? symbols (because removing those
        rows also completely removes the Never-worked value of the 'workclass' column from the dataset).
        """
        in_features = 104

        return (
            AdultNet(in_features=in_features)
            if not ternary
            else TernaryAdultNet(in_features, a, b)
        )
    elif data == "mushroom":
        in_features = 95
        return (
            AdultNet(in_features=in_features)
            if not ternary
            else TernaryAdultNet(in_features, a, b)
        )
    elif data == "logical_AND":
        in_features = 2
        return ANDNet(2)
    else:
        raise ValueError(f"Non-existing model: " + data)
