from mintorch.nn.module import Module
import mintorch.nn.functional as F 

class ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.ReLU.apply(x)

