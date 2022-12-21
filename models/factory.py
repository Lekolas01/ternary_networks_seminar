import torch.nn as nn

from models.lenet import LeNet, TernaryLeNet
from models.adult import AdultNet, TernaryAdultNet
from models.mushroom import MushroomNet, TernaryMushroomNet


def ModelFactory(data: str, ternary: bool, a: float=None, b: float=None) -> nn.Module:
    if (data == 'mnist'):
        n_classes  = 10
        return LeNet(n_classes) if not ternary else TernaryLeNet(n_classes, a, b)
    elif (data == 'adult'):
        in_features = 101
        return AdultNet(in_features=in_features) if not ternary else TernaryAdultNet(in_features, a, b)
    elif (data == 'mushroom'):
        in_features = 95
        return AdultNet(in_features=in_features) if not ternary else TernaryAdultNet(in_features, a, b)
    else:
        raise ValueError(f'Non-existing model: ' + data)
