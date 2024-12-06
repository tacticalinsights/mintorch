class Optimizer():

    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad = None
