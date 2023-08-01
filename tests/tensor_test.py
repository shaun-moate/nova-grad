## unit testing suite primarily having a better understanding of the components
import pytest, math, torch
from novagrad.tensor import Tensor

def test_engine_create_scalar():
    result = Tensor(6.0)
    assert result.data == 6.0 and result.grad == 0.0

def test_engine_add_scalars():
    a = Tensor(6.0)
    b = Tensor(-3.0)
    c = a + b
    assert c.data == 3.0

def test_engine_add_scalar_to_int():
    a = Tensor(6.0)
    b = 9 + a
    assert b.data == 15.0

def test_engine_add_scalar_to_float():
    a = Tensor(6.0)
    b = 5.0 + a
    assert b.data == 11.0

def test_engine_subtract_scalars():
    a = Tensor(6.0)
    b = Tensor(-3.0)
    c = a - b
    assert c.data == 9.0

def test_engine_subtract_scalar_to_int():
    a = Tensor(6.0)
    b = 9 - a
    assert b.data == 3.0

def test_engine_subtract_scalar_to_float():
    a = Tensor(6.0)
    b = 5.0 - a
    assert b.data == -1.0

def test_engine_multiply_scalars():
    a = Tensor(6.0)
    b = Tensor(-3.0)
    c = a * b
    assert c.data == -18.0

def test_engine_multiply_scalar_to_int():
    a = Tensor(6.0)
    b = 9 * a
    assert b.data == 54.0

def test_engine_multiply_scalar_to_float():
    a = Tensor(6.0)
    b = 5.0 * a
    assert b.data == 30.0

def test_engine_divide_scalars():
    a = Tensor(6.0)
    b = Tensor(-3.0)
    c = a / b
    assert c.data == -2.0

def test_engine_divide_scalar_to_int():
    a = Tensor(6.0)
    b = 9 / a
    assert b.data == 1.5

def test_engine_divide_scalar_to_float():
    a = Tensor(5.0)
    b = 4.0 / a
    assert b.data == 0.8

def test_engine_negate_scalars():
    a = Tensor(6.0)
    b = -a
    assert b.data == -6.0

def test_engine_power_scalars():
    a = Tensor(6.0)
    b = Tensor(2.0)
    with pytest.raises(AssertionError):
        c = a ** b
        print(c)

def test_engine_power_scalar_to_int():
    a = Tensor(2.0)
    b = a ** 3
    assert b.data == 8.0

def test_engine_power_scalar_to_float():
    a = Tensor(2.0)
    b = a ** 4.0
    assert b.data == 16.0

def test_engine_exp():
    a = Tensor(2.0)
    b = a.exp()
    assert b.data == math.exp(a.data)
    
def test_engine_tanh():
    a = Tensor(2.0)
    b = a.tanh()
    assert b.data == math.tanh(a.data)

def test_engine_forward_and_backward_pass():
    # using engine
    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # using pytorch
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_engine_forward_and_backward_pass_more_ops():
    # using engine
    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # using torch
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

