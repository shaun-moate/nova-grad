# A scalar property is a property that can be represented as a single value. 
# The properties that contain multiple values (either homogeneous or heterogeneous) are non-scalar properties.
class Tensor:
    def __init__(self, value, _children=(), _op="", label=""):
        self.data = value
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
        self._connections = set(_children)
        self._op = _op

    def __repr__(self): 
        return f"Tensor({self.data:.3f})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
        
    def __radd__(self, other): return self + other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    # TODO; review options for implementations of 'squashing' (activation functions) - ie. ReLu... or sigmoid (linear, non-linear)
    def relu(self):
        out = Tensor(max(0.0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Tensor(t, (self, ), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._connections:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()



