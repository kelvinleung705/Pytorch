from pkgutil import get_data
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
"""
batch_size = 64
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=2)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
"""

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            # First Convolutional Layer
            # Input: 1 channel (grayscale image), Output: 32 feature maps
            # Kernel size: 3x3
            # Padding: 1 pixel on each side to keep the image size 28x28
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max Pooling Layer: Reduces the image size by half (from 28x28 to 14x14)
            nn.MaxPool2d(kernel_size=2),

            # Second Convolutional Layer
            # Input: 32 channels (from previous layer), Output: 64 feature maps
            # Kernel size: 3x3
            # Padding: 1 pixel on each side to keep the image size 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max Pooling Layer: Reduces the image size by half again (from 14x14 to 7x7)
            nn.MaxPool2d(kernel_size=2)
        )

        # Now we flatten the output of the convolutional layers before feeding it to the linear layers.
        self.flatten = nn.Flatten()

        # The linear layers (classifier part) are the same as before, but the input size is different!
        # We need to calculate the size of the flattened output from the conv layers.
        # After two pooling layers, the image is 7x7. The last conv layer had 64 output channels.
        # So, the size is 64 channels * 7 pixels * 7 pixels = 3136.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # <-- The input size is now 3136, not 784!
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # 1. Pass the image through the convolutional layers
        # Input x shape: [64, 1, 28, 28]
        # Output x shape: [64, 64, 7, 7]
        x = self.conv_relu_stack(x)

        # 2. Flatten the output from the conv layers
        # Input x shape: [64, 64, 7, 7]
        # Output x shape: [64, 3136]
        x = self.flatten(x)

        # 3. Pass the correctly shaped vector into the linear layers
        # Input x shape: [64, 3136]
        # This will now work perfectly!
        logits = self.linear_relu_stack(x)
        return logits

"""
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
"""

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(device), y.to(device)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))


if __name__ == '__main__':
    batch_size = 64

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    loss_fn = nn.CrossEntropyLoss()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
    test_dataloader = WrappedDataLoader(test_dataloader, preprocess)

    model = NeuralNetwork().to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    epochs = 5

    fit(epochs, model, loss_fn, optimizer, train_dataloader, test_dataloader)

    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
