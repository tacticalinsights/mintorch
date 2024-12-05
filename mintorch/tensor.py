import numpy as np

import mintorch.autograd_engine as autograd_engine
import mintorch.nn.functional as F

class Tensor:
    
    def __init__(self, data, requires_grad=False, is_leaf=True, is_parameter=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.is_parameter = is_parameter
        self.grad_fn = None
        self.grad = None
        
    def __str__(self):
        if self.grad_fn:
            return f"{self.data}, grad_fn={self.grad_fn.__class__.__name__}"
        return f"{self.data}"
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx])
    
    # ------------------------------------------
    # ops that are NOT part of the comp graph
    # ------------------------------------------
    @property
    def shape(self):
        return self.data.shape
    
    def copy(self):
        return Tensor(self.data)
    
    def fill_(self, fill_value):
        self.data.fill(fill_value)
        
    @staticmethod
    def empty(*shape):
        return Tensor(np.empty(shape))
    
    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape))
    
    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape))
    
    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape))
    
    # ------------------------------------------
    # init backpropogation
    # ------------------------------------------
    def backward(self):
        output_grad = Tensor.ones(*self.data.shape)
        autograd_engine.backward(self.grad_fn, output_grad)
        
    # ------------------------------------------
    # ops that ARE part of the comp graph
    # ------------------------------------------
    def __neg__(self):
        return F.Neg.apply(self)
    
    def __add__(self, other):
        return F.Add.apply(self, other)
    
    def __sub__(self, other):
        return F.Sub.apply(self, other)
   
    def __mul__(self, other):
        return F.Mul.apply(self, other)
    
    def __truediv__(self, other):
        return F.Div.apply(self, other)
    
    def __matmul__(self, other):
        return F.MatMul.apply(self, other)
    
    def exp(self):
        return F.Exp.apply(self)
    
    def log(self):
        return F.Log.apply(self)
    
    def max(self, axis=None, keepdims=False):
        return F.Max.apply(self, axis, keepdims)
    
    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis, keepdims)
    
    def T(self):
        return F.Transpose.apply(self)
    
    def reshape(self, *shape):
        return F.Reshape.apply(self, shape)
