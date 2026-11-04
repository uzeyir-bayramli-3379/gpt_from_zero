import math
import random
class V:
    def __init__(self,data,prev=(),name=''):
        self.data,self.grad,self._prev,self.name=data,0.0,prev,name
        self._backward=lambda:None
    def __repr__(self):
        return f"V(data={self.data:.4f}, grad={self.grad:.4f})"
    def __add__(self, other):
        other = other if isinstance(other, V) else V(other)
        out=V(self.data+other.data,prev=(self,other))
        def _b():   
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward = _b
        return out
    def __mul__(self, other):
        other = other if isinstance(other,V) else V(other)
        out=V(self.data*other.data,prev=(self,other))
        def _b():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward =_b
        return out
    def __pow__(self, other):
        out = V(self.data**other,prev=(self,))
        def _b():
            self.grad+=self.data**(other-1)*other*out.grad
        out._backward = _b
        return out
    def tanh(self):
        out = V(math.tanh(self.data),prev=(self,))
        def _b():
            self.grad+=(1-out.data**2)*out.grad
        out._backward=_b
        return out
    def exp(self):
        out = V(math.exp(self.data),prev=(self,))
        def _b():
            self.grad+=out.data*out.grad
        out._backward=_b
        return out
    def log(self):
        out=V(math.log(self.data),prev=(self,))
        def _b():
            self.grad+=(1/self.data)*out.grad
        out._backward=_b
        return out
    def relu(self):
        out=V(max(0,self.data),prev=(self,))
        def _b():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward=_b
        return out
    def __truediv__(self, other): return self*(other**-1)
    def __rtruediv__(self, other): return other * (self**-1)
    def __neg__(self):         return self * -1              
    def __sub__(self, other):  return self + (-other)    
    def __radd__(self, other): return self + other   
    def __rmul__(self, other): return self * other   
    def __rsub__(self, other): return other + (-self)


class Neuron:
    def __init__(self,nin,act):
        self.w= list(V(random.uniform(-1, 1) / nin ** 0.5)for _ in range(nin)) 
        self.b=V(0.0)
        self.act=act
    def __call__(self, x):
        total=self.b
        for i in range(len(x)):
            total+=self.w[i]*x[i]
        return V.tanh(total) if self.act=='tanh' else (V.relu(total) if self.act=='relu' else total)
    def parameters(self):
        return self.w+[self.b]

class Layer:
    def __init__(self,nin,nout,act):
        self.neurons=[Neuron(nin,act) for _ in range(nout)]
    def __call__(self, x):
        out = []
        for n in self.neurons:
            out.append(n(x))
        return out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self,nin,nouts,act):
        self.layers=[]
        for i in range(len(nouts)):
            a = 'linear' if i == len(nouts) - 1 else act
            if i == 0:
                self.layers.append(Layer(nin,nouts[i],a))
            else:
                self.layers.append(Layer(nouts[i-1],nouts[i],a))
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]




def back_loop(root):
    deps,seen,stack={},{root},[root]
    while stack:
        using=stack.pop()
        for p in using._prev:
            if p not in seen:
                seen.add(p)
                stack.append(p)
            deps[p]=deps.get(p,0)+1
    root.grad=1.0
    ready=[root]
    while ready:
        using=ready.pop()
        using._backward()
        for p in using._prev:
            deps[p]=deps.get(p,0)-1
            if deps[p]==0:
                ready.append(p)

def zero_grad(params):
    for p in params:
        p.grad=0
def change(params,lr):
    for p in params:
        p.data-=lr*p.grad


def grad_check(params, loss_fn, eps=1e-6):
    for p in params: p.grad = 0.0
    loss = loss_fn()
    back_loop(loss)
    worst = 0
    for p in params:
        old = p.data
        p.data = old + eps; up   = loss_fn().data
        p.data = old - eps; down = loss_fn().data
        p.data = old
        numeric = (up - down) / (2 * eps)
        worst = max(worst, abs(p.grad - numeric))
    return worst





