import random
from engine import MLP,zero_grad,back_loop,change

def loss_fn():
    return sum((model(x)[0] - y) ** 2 for x, y in zip(xs, ys)) * (1 / len(xs))

random.seed(0)
model=MLP(2,[8,1],'tanh')
params=model.parameters()
xs = [[0,0], [0,1], [1,0], [1,1]]
ys = [-1,    1,     1,    -1] 

def train():
    for _ in range(350):
        zero_grad(params)
        loss = loss_fn()
        back_loop(loss)
        print(loss.data)
        change(params,0.1)
    for x, y in zip(xs, ys):
        print(x, "target:", y, "pred:", round(model(x)[0].data, 3))

if __name__ == "__main__":
    train()
