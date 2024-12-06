from mintorch.nn.module import Module

class Sequential(Module):

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

        for idx, layer in enumerate(self.layers):
            self.add_module(str(idx), layer)

    def __iter__(self):
        yield from self.layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
