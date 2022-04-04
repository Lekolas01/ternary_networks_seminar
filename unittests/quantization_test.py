import unittest
import torch
import torch.nn as nn

from models.ternary import TernaryLinear, TernaryModule


class TestSimplification(unittest.TestCase):
    class TestModel(TernaryModule):
        def __init__(self, bias: bool, fill_value=None):
            classifier = nn.Sequential(
                TernaryLinear(in_features=5, out_features=5, bias=bias),
                TernaryLinear(in_features=5, out_features=5, bias=bias)
            )
            super().__init__(classifier, 0, 0)
            self.fill_weight(fill_value)
            self.fill_bias(fill_value)


        def fill_params(self, fill_value: float):
            if fill_value == None: return

            for param in self.classifier.parameters():
                param.data.fill_(fill_value)

        def fill_attr(self, attr_value: float, attr_name: str):
            with torch.no_grad():
                for layer in self.classifier:
                    if hasattr(layer, attr_name) and getattr(layer, attr_name) is not None:
                        getattr(layer, attr_name).fill_(attr_value)
        
        def fill_weight(self, value: float):
            self.fill_attr(value, 'weight')
        
        def fill_bias(self, value:float):
            self.fill_attr(value, 'bias')

    
    def get_sample_input(self, n_samples = 100):
        return torch.rand((n_samples, 5))


    def test_fully_connected(self):
        """Bias enabled and all weights and bias parameters set to 1."""
        model = self.TestModel(bias=True, fill_value=0.7)
        q_model = model.quantized(simplify=False)
        simple_model = model.quantized(simplify=True)
        
        for idx, layer in enumerate(q_model.classifier):
            assert(
                simple_model.classifier[idx].weight.shape 
                == model.classifier[idx].weight.shape 
                == q_model.classifier[idx].weight.shape)
            if (layer.bias is None): continue
            assert(
                simple_model.classifier[idx].bias.shape 
                == model.classifier[idx].bias.shape 
                == q_model.classifier[idx].bias.shape)
            
            input = torch.rand((100, 5))
            assert(torch.allclose(q_model(input)[0], simple_model(input)[0]))


    def test_full_connected_no_bias(self):
        """Bias is disabled and therefore None."""
        model = self.TestModel(bias=False, fill_value=0.7)
        q_model = model.quantized(simplify=False)
        simple_model = model.quantized(simplify=True)

        for idx, layer in enumerate(model.classifier):
            assert(q_model.classifier[idx].weight.shape == simple_model.classifier[idx].weight.shape == layer.weight.shape)
            assert(model.classifier[idx].bias is None)
            assert(q_model.classifier[idx].bias is None)
            assert(simple_model.classifier[idx].bias is None)

        input = torch.rand((100, 5))
        assert(torch.allclose(q_model(input)[0], simple_model(input)[0]))
    

    def test_full_connected_zero_bias(self):
        """Bias is enabled, but set to 0. Other weights set to 1."""
        model = self.TestModel(bias=True, fill_value=0.7)
        model.fill_bias(0.0)
        q_model = model.quantized(simplify=False)
        simple_model = model.quantized(simplify=True)

        for idx, layer in enumerate(model.classifier):
            assert(model.classifier[idx].weight.shape == q_model.classifier[idx].weight.shape == simple_model.classifier[idx].weight.shape)
            assert(model.classifier[idx].bias is not None)
            assert(q_model.classifier[idx].bias is not None)
            assert(simple_model.classifier[idx].bias is None)
    
        input = torch.rand((100, 5))
        assert(torch.allclose(q_model(input)[0], simple_model(input)[0]))
    

    def test_output_size(self):
        """Assert that you don't cut out nodes in the output of the last layer."""
        model = self.TestModel(bias=False, fill_value=0.0)
        
        w1 = torch.tensor(
            [[0, 0, 0, 0, 0],
             [0, 1, 2, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 3, 0],
             [0, 0, 0, 0, 0]])

        w2 = torch.tensor(
            [[1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]])

        with torch.no_grad():
            model.classifier[0].weight[:] = w1
            model.classifier[1].weight[:] = w2

        q_model = model.quantized(simplify=False)
        simple_model = model.quantized(simplify=True)

        for idx, layer in enumerate(q_model.classifier):
            assert(model.classifier[idx].weight.shape == model.classifier[idx].weight.shape == layer.weight.shape)

        input = torch.rand((100, 5))
        q_out = q_model(input)[0]
        simple_out = simple_model(input)[0]
        assert(torch.allclose(q_out, simple_out))


    def test_empty_then_full(self):
        """First layer (i.e. weight and bias) only zeros, second layer only ones. This should return exactly the bias in the second layer of the quantized network."""
        model = self.TestModel(bias=True, fill_value=0.0)

        w2 = torch.ones_like(model.classifier[1].weight)
        b2 = torch.ones_like(model.classifier[1].bias)

        with torch.no_grad():
            model.classifier[1].weight[:] = w2
            model.classifier[1].bias[:] = b2

        q_model = model.quantized(simplify=False)
        simple_model = model.quantized(simplify=True)

        for idx, layer in enumerate(q_model.classifier):
            assert(model.classifier[idx].weight.shape == model.classifier[idx].weight.shape == layer.weight.shape)

        input = torch.rand((100, 5))
        q_out = q_model(input)[0]
        simple_out = simple_model(input)[0]
        assert(torch.allclose(q_out, simple_out))

