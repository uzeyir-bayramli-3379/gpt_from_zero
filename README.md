# gpt_from_zero

A small deep-learning framework built from scratch in pure Python, with the goal of training a tiny GPT on top of it. No NumPy, no PyTorch: only the standard library.

The idea is to understand the "assembly language" of machine learning. Frameworks like PyTorch hide automatic differentiation, layers and the training loop behind a few lines of code. Building each piece by hand, and checking every gradient numerically, shows what those lines actually do.

**Status: work in progress.** The scalar autograd engine, a multilayer perceptron and the tests are done. The transformer is still ahead (see the roadmap below).

## What's here

| File | Contents |
|---|---|
| `engine.py` | The `V` autograd value, `Neuron`, `Layer`, `MLP`, the backward pass (`back_loop`), `zero_grad`, `change` (SGD update) and `grad_check` |
| `train_xor.py` | Trains a small MLP on XOR |
| `test_grad.py` | Gradient checks for every operation and for whole networks |

## Quick start

Requires Python 3 and nothing else.

```bash
python train_xor.py     # trains XOR and prints the loss and final predictions
python test_grad.py     # runs all gradient checks (raises an error if one fails)
```

## What the engine supports

- **Operations:** add, subtract, multiply, divide, power, negate, `exp`, `log`, `tanh`, `relu`
- **Layers:** `Neuron`, `Layer` and `MLP`, with a per-layer activation (`tanh`, `relu` or `linear`); the output layer is always linear
- **Training:** mean squared error loss, plain gradient descent (`p.data -= lr * p.grad`)

## How it works

Every number is a `V` object that stores its value, its gradient, the values it was computed from, and a small closure that knows its own local derivative. Running a forward pass builds a computation graph as a side effect.

The backward pass (`back_loop`) visits the graph in an order that is safe for the chain rule:

1. Walk the graph once and count, for every node, how many other nodes consume it.
2. Start from the loss with gradient 1.
3. Process a node only after all of its consumers have passed their gradient to it (the count reaches zero), then push the gradient into its inputs with `+=`.

This is the same dependency-counting idea PyTorch's autograd uses. It is iterative, so it does not hit Python's recursion limit on deep graphs, and the `+=` is what makes a weight that is used in many places (or by many training examples) collect the sum of all of their contributions.

## Testing

Backpropagation bugs do not crash anything; the network just trains badly. So every operation and every network is verified against a numerical gradient (central differences):

```
numeric slope = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

`test_grad.py` asserts that backprop and the numerical slope agree to within `1e-8`. The current worst-case differences are around `1e-10` for every operation and for both a `tanh` and a `relu` network. One thing worth knowing: ReLU has a corner at 0, so the check can report a false failure when a pre-activation sits exactly at 0. The test therefore uses nonzero biases for the ReLU network.

## Results

XOR with `MLP(2, [8, 1], 'tanh')`, learning rate 0.1, 350 steps:

```
loss: 1.0076 -> 1.5e-08
[0, 0] target: -1  pred: -1.0
[0, 1] target:  1  pred:  1.0
[1, 0] target:  1  pred:  1.0
[1, 1] target: -1  pred: -1.0
```

## Limitations

- Scalar by design: every number is a separate Python object, which makes the code easy to read and slow to run. Training a network with about a thousand parameters takes several seconds per thousand examples. Practical sizes need the tensor version on the roadmap.
- Only gradient descent so far, with no batching, no Adam and no GPU.

## Roadmap

- [x] Scalar autograd engine with iterative dependency-counted backward pass
- [x] Neurons, layers, MLP, per-layer activations
- [x] Gradient checks for every operation and network
- [x] XOR training
- [ ] Softmax and cross-entropy loss
- [ ] Classifier on real data (hand-gesture landmarks)
- [ ] Tensor version of the engine on NumPy (matrix multiply, broadcasting, reshape)
- [ ] `Embedding`, `LayerNorm`, GELU, Adam
- [ ] Causal self-attention and a transformer block
- [ ] Character-level tiny GPT: train it and generate text
