from trying import V
import random
random.seed(0)
class Neuron:
    def __init__(self,nin):
        self.w= list(V(random.uniform(-1, 1) / nin ** 0.5)for _ in range(nin)) 
        self.b=V(0.0)
    def __call__(self, x):
        total=self.b
        for i in range(len(x)):
            total+=self.w[i]*x[i]
        return V.tanh(total)
    def parameters(self):
        return self.w+[self.b]

class Layer:
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        out = []
        for n in self.neurons:
            out.append(n(x))
        return out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self,nin,nouts):
        self.layers=[]
        for i in range(len(nouts)):
            if i == 0:
                self.layers.append(Layer(nin,nouts[i]))
            else:
                self.layers.append(Layer(nouts[i-1],nouts[i]))
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

