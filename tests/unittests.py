import unittest
import numpy as np

import torch
import torch.nn as nn

from models.lenet5 import TernaryLinear, TernaryConv2d


class TestTernaryLayers(unittest.TestCase):
    def test_ternary_linear(self):
        def test_for_bias(bias):
            x = torch.zeros(size=(5, 5))
            x[2, 2] = 1
            ternary_linear = TernaryLinear(in_features=5, out_features=8, bias=bias)
            nn.init.zeros_(ternary_linear.weight)
            if (ternary_linear.bias is not None): 
                nn.init.zeros_(ternary_linear.bias)
            with torch.no_grad():
                ternary_linear.weight[2, 2] = 1
            y = ternary_linear(x)

            self.assertAlmostEqual(torch.sum(y).item(), np.tanh(1))
            new_y = ternary_linear(x) # also check for idempotence
            self.assertAlmostEqual(torch.sum(new_y).item(), np.tanh(1))
        
        test_for_bias(False)
        test_for_bias(True)

        
    def test_ternary_conv2d(self):
        def test_for_bias(bias):
            x = torch.zeros(size=(10, 1, 5, 5))
            x[0, 0, 2, 2] = 1
            ternary_conv2d = TernaryConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=bias)
            nn.init.zeros_(ternary_conv2d.weight)
            if (ternary_conv2d.bias is not None): 
                nn.init.zeros_(ternary_conv2d.bias)
            with torch.no_grad():
                ternary_conv2d.weight[0, 0, 2, 2] = 1
            y = ternary_conv2d(x)

            self.assertAlmostEqual(torch.sum(y).item(), np.tanh(1))
            new_y = ternary_conv2d(x) # also check for idempotence
            self.assertAlmostEqual(torch.sum(new_y).item(), np.tanh(1))
        test_for_bias(False)
        test_for_bias(True)


if __name__ == '__main__':
    unittest.main()