import mintorch.nn.functional as F 

class CrossEntropyLoss:

    def __call__(self, predicted, target):
        return self.forward(predicted, target)

    @staticmethod
    def forward(predicted, target):
        return F.cross_entropy(predicted, target)
