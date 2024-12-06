import numpy as np

import mintorch.tensor as tensor
from mintorch.autograd_engine import Function

def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Neg(Function):
    
    @staticmethod
    def forward(ctx, a):
        data = -a.data
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a_grad = output_grad.data * -1
        return tensor.Tensor(a_grad), None
    
class Add(Function):
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        data = a.data + b.data
        requires_grad = a.requires_grad + b.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, b = ctx.saved_tensors
        a_grad = output_grad.data * 1
        b_grad = output_grad.data * 1
        return (tensor.Tensor(unbroadcast(a_grad, a.shape)),
                tensor.Tensor(unbroadcast(b_grad, b.shape)))
        
class Sub(Function):
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        data = a.data - b.data
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, output_grad):
        a, b = ctx.saved_tensors
        a_grad = output_grad.data * 1
        b_grad = output_grad.data * -1
        return (tensor.Tensor(unbroadcast(a_grad, a.shape)), 
                tensor.Tensor(unbroadcast(b_grad, b.shape)))
        
class Mul(Function):
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, b = ctx.saved_tensors
        a_grad = output_grad.data * b.data
        b_grad = output_grad.data * a.data
        return (tensor.Tensor(unbroadcast(a_grad, a.shape)), 
                tensor.Tensor(unbroadcast(b_grad, b.shape)))
        
class Div(Function):
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        data = a.data / b.data
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, b = ctx.saved_tensors
        a_grad = output_grad.data * (1 / b.data)
        b_grad = output_grad.data * (-a.data / b.data ** 2)
        return (tensor.Tensor(unbroadcast(a_grad, a.shape)), 
                tensor.Tensor(unbroadcast(b_grad, b.shape)))
        
class MatMul(Function):
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, b = ctx.saved_tensors
        a_grad = output_grad.data @ np.swapaxes(b.data, -1, -2)
        b_grad = np.swapaxes(a.data, -1, -2) @ output_grad.data
        return (tensor.Tensor(unbroadcast(a_grad, a.shape)), 
                tensor.Tensor(unbroadcast(b_grad, b.shape)))
        
class Exp(Function):
    
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        data = np.exp(a.data)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, = ctx.saved_tensors
        a_grad = output_grad.data * np.exp(a.data)
        return tensor.Tensor(a_grad), None
    
class Log(Function):
    
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        data = np.log(a.data)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, output_grad):
        a, = ctx.saved_tensors
        a_grad = output_grad.data * (1 / a.data)
        return tensor.Tensor(a_grad), None
    
class Max(Function):
    
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims=keepdims
        data = np.max(a.data, axis=axis, keepdims=keepdims)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        if axis is not None and not keepdims:
            output_grad = np.expand_dims(output_grad.data, axis=axis)
        mask = (a.data == np.max(a.data, axis=axis, keepdims=True))#.astype(float)
        a_grad = output_grad.data * mask
        assert(a_grad.shape == a.shape)
        return tensor.Tensor(a_grad), None
    
class Sum(Function):
    
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        data = np.sum(a.data, axis=axis, keepdims=keepdims)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        shape = ctx.shape
        axis = ctx.axis
        keepdims = ctx.keepdims
        if axis is not None and not keepdims:
            output_grad = np.expand_dims(output_grad.data, axis=axis)
        a_grad = np.ones(shape) * output_grad.data
        assert(a_grad.shape == shape)
        return tensor.Tensor(a_grad), None
    
class Transpose(Function):
    
    @staticmethod
    def forward(ctx, a):
        data = a.data.T
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a_grad = output_grad.data.T
        return tensor.Tensor(a_grad), None
    
class Reshape(Function):
    
    @staticmethod
    def forward(ctx, a, shape):
        ctx.shape = a.shape
        data = a.data.reshape(shape)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        shape = ctx.shape
        a_grad = output_grad.data.reshape(shape)
        return tensor.Tensor(a_grad), None
    
class ReLU(Function):
    
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        data = a.data * (a.data > 0)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensor.Tensor(data, requires_grad, is_leaf)
    
    @staticmethod
    def backward(ctx, output_grad):
        a, = ctx.saved_tensors
        a_grad = output_grad.data * (a.data > 0)
        return tensor.Tensor(a_grad), None
    
def cross_entropy(predicted, target):
    batch_size, num_classes = predicted.shape
    logits = predicted - predicted.max(axis=1, keepdims=True)
    exp_logits = logits.exp()
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = probs.log() * to_one_hot(target, num_classes)
    cost = -loss.sum() / tensor.Tensor(batch_size)
    return cost
    
def to_one_hot(arr, num_classes):
    arr = arr.data.astype(int)
    data = np.zeros((arr.shape[0], num_classes))
    data[np.arange(arr.shape[0]), arr] = 1
    return tensor.Tensor(data, requires_grad=True)