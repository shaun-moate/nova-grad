from novagrad.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_inputs: int, method: str):
        import random
        self.weights = [Tensor(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.bias = Tensor(random.uniform(-1,1))
        self.method = method

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return act.tanh() if self.method == 'tanh' else act.relu()

    def parameters(self) -> list[Tensor]:
        return self.weights + [self.bias]

    def __repr__(self):
        return f"Neuron({len(self.weights)})"

class Layer(Module):
    def __init__(self, n_inputs: int, n_outputs: int, method: str):
        self.neurons = [Neuron(n_inputs, method) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MultiLayerPerceptron(Module):
    def __init__(self, n_inputs: int, n_outputs: list, method: str):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1], method) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
