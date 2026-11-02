import math

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
    def __neg__(self):         return self * -1              
    def __sub__(self, other):  return self + (-other)    
    def __radd__(self, other): return self + other   
    def __rmul__(self, other): return self * other   
    def __rsub__(self, other): return other + (-self)

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
def forward():
    m1=w1*x
    z1=m1+b1
    h1=V.tanh(z1)

    m2=w2*x
    z2=m2+b2
    h2=V.tanh(z2)

    pred = h1+h2

    return (pred - target) ** 2
lr=0.1

x=V(2.0)
w1=V(0.5)
b1=V(-1.0)
w2=V(0.25)
b2=V(0.5)

m1=w1*x
z1=m1+b1
h1=V.tanh(z1)

m2=w2*x
z2=m2+b2
h2=V.tanh(z2)

pred=h1+h2
target=1.5

params = [w1, b1, w2, b2] 

#for p in params:
#    print( p.data)
for i in range(10):
    zero_grad(params)
    loss=forward()
    back_loop(loss)
    print(loss.data)
    change(params,0.1)
