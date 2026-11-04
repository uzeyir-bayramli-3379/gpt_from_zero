import random
from engine import V, MLP, grad_check
from train_xor import params, loss_fn, xs, ys

TOL = 1e-8

def check(name, ps, fn):
    err = grad_check(ps, fn)
    print(f"{name:10s} {err:.2e}")
    assert err < TOL, f"{name} gradient check failed: {err}"

def test_ops():
    x = V(0.7)
    check("exp", [x], lambda: x.exp())
    check("log", [x], lambda: x.log()) 
    check("relu +", [x], lambda: x.relu())
    y = V(-0.7)
    check("relu -", [y], lambda: y.relu())
    a, b = V(1.3), V(0.6)
    check("div", [a, b], lambda: a / b)
    check("rdiv", [b], lambda: 2 / b)
    check("sigmoid", [x], lambda: 1 / (1 + (-x).exp()))
    check("tanh +", [x], lambda: x.tanh())
    check("tanh -", [y], lambda: y.tanh())

def test_tanh_network():
    check("net tanh", params, loss_fn)

def test_relu_network():
    random.seed(0)
    model = MLP(2, [8, 1], 'relu')
    for layer in model.layers:
        for n in layer.neurons:
            n.b.data = random.uniform(-0.5, 0.5)
    def fn():
        return sum((model(x)[0] - y) ** 2 for x, y in zip(xs, ys)) * (1 / len(xs))
    check("net relu", model.parameters(), fn)

if __name__ == "__main__":
    test_ops()
    test_tanh_network()
    test_relu_network()
    print("with thoughts of failure sinking in")
