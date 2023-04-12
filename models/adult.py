import torch.nn as nn

from .ternary import TernaryModule, TernaryLinear


class AdultNet(nn.Module):
    def __init__(self, in_features=104, **kwargs):
        """You get exactly 104 input features if you one hot encode all independent variables and 
        dummy encode the target variable AFTER removing the rows with ? symbols (because removing those 
        rows also completely removes the Never-worked value of the 'workclass' column from the dataset).
        """
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ans = self.classifier(x).flatten()
        return ans, ans


class TernaryAdultNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=80),
            nn.Sigmoid(),
            TernaryLinear(in_features=80, out_features=1),
            nn.Sigmoid()
        )
        super().__init__(classifier, a, b, **kwargs)

