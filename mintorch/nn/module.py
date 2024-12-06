# https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module 

from mintorch.tensor import Tensor

class Module:

    def __init__(self):
        self._parameters = {}
        self._submodules = {}

        self.is_train = True
    
    def add_parameter(self, name, value):
        self._ensure_is_initialized()
        self._parameters[name] = value

    def add_module(self, name, value):
        self._ensure_is_initialized()
        self._submodules[name] = value

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.is_parameter:
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)
    
    def parameters(self):
        self._ensure_is_initialized()
        for _, parameter in self._parameters.items():
            yield parameter
        for _, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter

    def train(self):
        self.is_train = True
        for submodule in self._submodules.values():
            submodule.train()

    def eval(self):
        self.is_train = False
        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args):
        return self.forward(*args)

    def _ensure_is_initialized(self):
        if "_submodules" not in self.__dict__:
            raise Exception("Module not initialized.",
                            "Did you forget to call super().__init__()?")
