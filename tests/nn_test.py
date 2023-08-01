import pytest
from novagrad.tensor import Tensor
from novagrad.nn import Neuron, Layer, MultiLayerPerceptron

def test_neuron_as_expected():
    n = Neuron(3, "relu")
    np = n.parameters()
    assert isinstance(np, list) == True and isinstance(np[0], Tensor) == True

def test_neuron_is_callable():
    n = Neuron(3, "relu")
    xs = n([1.0, -2.0, 5.0])
    assert isinstance(xs, Tensor) == True

def test_layer_as_expected():
    l = Layer(3, 3, "relu")
    lp = l.parameters()
    assert isinstance(lp, list) == True

def test_layer_is_callable():
    l = Layer(3, 3, "relu")
    xs = l([1.0, -2.0, 5.0])
    assert isinstance(xs, list) == True

def test_mlp_as_expected():
    mlp = MultiLayerPerceptron(3, [3, 3, 1], "relu")
    mlpp = mlp.parameters()
    assert isinstance(mlpp, list) == True

def test_mlp_is_callable():
    mlp = MultiLayerPerceptron(3, [3, 3, 1], "relu")
    xs = mlp([1.0, -2.0, 5.0])
    assert isinstance(xs, Tensor) == True

def test_mlp_is_callable_multi_outputs():
    mlp = MultiLayerPerceptron(3, [3, 3, 5], "relu")
    xs = mlp([1.0, -2.0, 5.0])
    assert isinstance(xs, list) == True
