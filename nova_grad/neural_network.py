import random
from nova_grad.engine import Scalar

class Base:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Base):
    def __init__(self, n_inputs: int):
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.bias = Scalar(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return act.tanh()

    def parameters(self) -> list[Scalar]:
        return self.weights + [self.bias]

    def __repr__(self):
        return f"TanhNeuron({len(self.weights)})"

class Layer(Base):
    def __init__(self, n_inputs: int, n_outputs: int):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MultiLayerPerceptron(Base):
    def __init__(self, n_inputs: int, n_outputs: list):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
