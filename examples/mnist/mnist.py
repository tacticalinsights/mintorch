import os
import sys
sys.path.append("../../")

import numpy as np 

import mintorch.tensor as tensor
from mintorch.nn.sequential import Sequential
from mintorch.nn.linear import Linear
from mintorch.nn.activations import ReLU
from mintorch.nn.loss import CrossEntropyLoss
from mintorch.optim.sgd import SGD

def load_data(root="./data/", batch_size=100):

    # read
    train_x = np.load((os.path.join(root, "train_data.npy")))
    train_y = np.load((os.path.join(root, "train_labels.npy")))
    val_x = np.load((os.path.join(root, "val_data.npy")))
    val_y = np.load((os.path.join(root, "val_labels.npy")))
    train_m, val_m = train_x.shape[0], val_x.shape[0]

    # to tensor
    train_x = tensor.Tensor(train_x)
    train_y = tensor.Tensor(train_y)
    val_x = tensor.Tensor(val_x)
    val_y = tensor.Tensor(val_y)

    # normalize
    train_x = train_x / tensor.Tensor(255)
    val_x = val_x / tensor.Tensor(255)

    # shuffle
    indices = np.arange(train_m)
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    # batch
    train_data = [(train_x[i: i + batch_size], train_y[i: i + batch_size])
                  for i in range(0, train_m, batch_size)
                 ]
    val_data = [(val_x[i: i + batch_size], val_y[i: i + batch_size])
                for i in range(0, val_m, batch_size)
               ]

    return train_data, val_data

def train(model, criterion, optimizer, train_data, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_data):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1} / {num_epochs}],",
                      f"Step [{i + 1} / {len(train_data)}]",
                      f"Loss: {loss.data:.4f}")

def val(model, val_data):
    num_correct = 0
    model.eval()
    for (data, labels) in val_data:
        logits = model(data)
        preds = np.argmax(logits.data, axis=1)
        num_correct += np.sum(preds == labels.data)

    batch_size = val_data[0][1].shape[0]
    print(f"Accuracy: {num_correct / (len(val_data) * batch_size):.4f}")

if __name__ == '__main__':
    train_data, val_data = load_data()
    model = Sequential(
                Linear(784, 20),
                ReLU(),
                Linear(20, 10),
            )
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    train(model, criterion, optimizer, train_data)
    print()
    val(model, val_data)
