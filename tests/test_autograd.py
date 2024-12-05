import sys
sys.path.append('../')

import torch
import numpy as np
np.seterr(all='ignore')

from mintorch.tensor import Tensor
from mintorch.autograd_engine import *
from mintorch.nn.functional import *

def main():
    test_neg()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_matmul()
    test_exp()
    test_log()
    test_max()
    test_sum()
    test_transpose()
    test_reshape()
    test_relu()
    test1()
    test2()

# ------------------------------------------
# functional.py tests
# ------------------------------------------

def test_neg():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Neg.forward(ctx, a)
    c_torch = -a_torch
    
    # back prop
    grads = Neg.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_add():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    
    # forward prop
    ctx = ContextManager()
    c = Add.forward(ctx, a, b)
    c_torch = a_torch + b_torch
    
    # back prop
    grads = Add.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    assert check_val_and_grad(grads[1], b_torch.grad)
    
def test_sub():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    
    # forward prop
    ctx = ContextManager()
    c = Sub.forward(ctx, a, b)
    c_torch = a_torch - b_torch
    
    # back prop
    grads = Sub.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    assert check_val_and_grad(grads[1], b_torch.grad)
    
def test_mul():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    
    # forward prop
    ctx = ContextManager()
    c = Mul.forward(ctx, a, b)
    c_torch = a_torch * b_torch
    
    # back prop
    grads = Mul.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    assert check_val_and_grad(grads[1], b_torch.grad)
    
def test_div():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    
    # forward prop
    ctx = ContextManager()
    c = Div.forward(ctx, a, b)
    c_torch = a_torch / b_torch
    
    # back prop
    grads = Div.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    assert check_val_and_grad(grads[1], b_torch.grad)
    
def test_matmul():
    
    # input tensors
    a_shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*a_shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    b_shape = a_shape[:-2] + (a_shape[-1], a_shape[-2])
    b = Tensor.randn(*b_shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    
    # forward prop
    ctx = ContextManager()
    c = MatMul.forward(ctx, a, b)
    c_torch = a_torch @ b_torch
    
    # back prop
    grads = MatMul.backward(ctx, Tensor.ones(*tuple(c_torch.shape)))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    assert check_val_and_grad(grads[1], b_torch.grad)
    
def test_exp():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Exp.forward(ctx, a)
    c_torch = a_torch.exp()
    
    # back prop
    grads = Exp.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_log():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Log.forward(ctx, a)
    c_torch = a_torch.log()
    
    # back prop
    grads = Log.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_max(axis=None, keepdims=False):
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    axis = np.random.randint(0, len(shape)) if axis else axis
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Max.forward(ctx, a, axis=axis, keepdims=keepdims)
    c_torch = a_torch.amax(axis=axis, keepdims=keepdims)
    
    # back prop
    grads = Max.backward(ctx, Tensor.ones(*tuple(c_torch.shape)))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_sum(axis=None, keepdims=False):
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    axis = np.random.randint(0, len(shape)) if axis else axis
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Sum.forward(ctx, a, axis=axis, keepdims=keepdims)
    c_torch = a_torch.sum(axis=axis, keepdims=keepdims)
    
    # back prop
    grads = Sum.backward(ctx, Tensor.ones(*tuple(c_torch.shape)))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_transpose():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 3)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Transpose.forward(ctx, a)
    c_torch = a_torch.T
    
    # back prop
    grads = Transpose.backward(ctx, Tensor.ones(*tuple(c_torch.shape)))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_reshape():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    new_shape = tuple(np.random.permutation(shape))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = Reshape.forward(ctx, a, new_shape)
    c_torch = a_torch.reshape(new_shape)
    
    # back prop
    grads = Reshape.backward(ctx, Tensor.ones(*tuple(c_torch.shape)))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
def test_relu():
    
    # input tensors
    shape = tuple(np.random.randint(1, 9, np.random.randint(2, 5)))
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    # forward prop
    ctx = ContextManager()
    c = ReLU.forward(ctx, a)
    c_torch = a_torch.relu()
    
    # back prop
    grads = ReLU.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()
    
    # checks
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(grads[0], a_torch.grad)
    
# ------------------------------------------
# general autograd tests
# ------------------------------------------
    
def test1():
    a = Tensor.randn(2, 6)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    b = Tensor.randn(2, 6)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)
    c = Tensor.randn(2, 6)
    c.requires_grad = True
    c_torch = get_same_torch_tensor(c)
    d = Tensor.randn(2, 6)
    d.requires_grad = True
    d_torch = get_same_torch_tensor(d)
    
    e = (a.log() + b * c) / d.exp()
    e_torch = (a_torch.log() + b_torch * c_torch) / d_torch.exp()
    
    e.backward()
    e_torch.sum().backward()
    
    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(e, e_torch)
    
def test2():
    a = Tensor.randn(2, 3, 4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)
    
    b = a.reshape(2, 4, 3)
    b_torch = a_torch.reshape(2, 4, 3)
    c = b.sum(axis=1, keepdims=False)
    c_torch = b_torch.sum(axis=1, keepdims=False)
    d = c.max(axis = 0, keepdims=True)
    d_torch = c_torch.amax(axis=0, keepdims=True)
    e = d.T()
    e_torch = d_torch.T
    
    e.backward()
    e_torch.sum().backward()
    
    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(e, e_torch)

# ------------------------------------------
# helpers
# ------------------------------------------

def get_same_torch_tensor(mintorch_tensor):
    out = torch.tensor(mintorch_tensor.data).double()
    out.requires_grad = mintorch_tensor.requires_grad
    return out
    
def check_val_and_grad(mintorch_tensor, pytorch_tensor, difference_threshold=1e-10):
    return check_val(mintorch_tensor, pytorch_tensor, difference_threshold) and \
           check_grad(mintorch_tensor, pytorch_tensor, difference_threshold)

def check_val(mintorch_tensor, pytorch_tensor, difference_threshold=1e-8):
    if mintorch_tensor.shape != tuple(pytorch_tensor.shape):
        print(
            "mintorch tensor and pytorch tensor have different shapes:",
            f"{mintorch_tensor.shape}, {pytorch_tensor.shape}"
        )
        return False
    
    difference = np.abs(mintorch_tensor.data - pytorch_tensor.data.numpy())
    max_difference = np.nanmax(difference)
    if np.isnan(max_difference):
        max_difference = 0
        
    if max_difference < difference_threshold:
        return True
    else:
        print(f"the given tensors differ by at least {max_difference}")
        print(f"mintorch tensor:\n{mintorch_tensor}")
        print(f"pytorch tensor:\n{pytorch_tensor}")
        return False
      
def check_grad(mintorch_tensor, pytorch_tensor, difference_threshol=1e-10):
    if mintorch_tensor.grad is None and pytorch_tensor_no_grad(pytorch_tensor):
        return True
    elif mintorch_tensor.grad is None:
        print('mintorch tensor grad is none, but pytorch tensor grad is not')
        return False
    elif pytorch_tensor_no_grad(pytorch_tensor):
        print('pytorch tensor grad is none, but mintorch tensor grad is not')
        return False
    else:
        return check_val(mintorch_tensor.grad, pytorch_tensor.grad)
    
# unaffected by retain_grad set to True
def pytorch_tensor_no_grad(pytorch_tensor):
    return not pytorch_tensor.requires_grad \
        or not pytorch_tensor.is_leaf \
        or pytorch_tensor.grad is None

if __name__ == "__main__":
    main()
