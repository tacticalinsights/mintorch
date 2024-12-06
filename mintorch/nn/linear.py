import numpy as np 

from mintorch.nn.module import Module
from mintorch.tensor import Tensor

class Linear(Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight_data = np.random.rand(out_features, in_features) * np.sqrt(2 / in_features)
        self.weight = Tensor(weight_data, requires_grad=True, is_parameter=True)
        self.bias = Tensor.zeros(out_features, requires_grad=True, is_parameter=True)

    def forward(self, x):
        return x @ self.weight.T() + self.bias
