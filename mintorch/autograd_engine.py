from mintorch import tensor

def backward(grad_fn, output_grad):
    input_grads = grad_fn.apply(output_grad)
    for i, parent in enumerate(grad_fn.next_functions):
        if isinstance(parent, BackwardFunction):
            backward(parent, input_grads[i])
        elif isinstance(parent, AccumulateGrad):
            parent.apply(input_grads[i])

class Function:
    
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("All subclasses must implement forward")
        
    @staticmethod
    def backward(ctx, *output_grads):
        raise NotImplementedError("All subclasses must implement backward")
        
    @classmethod
    def apply(cls, *args):
        backward_function = BackwardFunction(cls)
        output_tensor = cls.forward(backward_function.ctx, *args)
        for arg in args:
            if isinstance(arg, tensor.Tensor):
                if arg.is_leaf and arg.requires_grad: # parameter
                    arg.grad_fn = AccumulateGrad(arg)
                    backward_function.next_functions.append(arg.grad_fn)
                elif arg.requires_grad:
                    backward_function.next_functions.append(arg.grad_fn)
                else:
                    backward_function.next_functions.append(None)
        output_tensor.grad_fn = backward_function
        return output_tensor

class AccumulateGrad:
    
    def __init__(self, tensor):
        self.tensor = tensor
        self.function_name = "AccumulateGrad"
        
    def apply(self, grad):
        if self.tensor.grad is None:
            self.tensor.grad = tensor.Tensor(grad.data)
        else:
            self.tensor.grad.data += grad.data

class BackwardFunction:
    
    def __init__(self, cls):
        self.ctx = ContextManager()
        self._forward_cls = cls
        self.next_functions = []
        self.funcntion_name = cls.__name__
        
    def apply(self, *args):
        return self._forward_cls.backward(self.ctx, *args)

class ContextManager:
    
    def __init__(self):
        self.saved_tensors = []
        
    def save_for_backward(self, *args):
        for arg in args:
            if type(arg).__name__ != "Tensor":
                raise Exception(
                    f"Expected objects of type 'Tensor', but received type",
                    f"'{type(arg).__name__}'"
                )
            self.saved_tensors.append(arg.copy())
