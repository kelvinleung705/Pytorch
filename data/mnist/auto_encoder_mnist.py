import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Hyperparameters ---
EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
LATENT_DIM = 8  # The size of our compressed representation

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a transform to normalize the data
transform = transforms.ToTensor()

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Download and load the test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        # --- Encoder ---
        # It will take a flattened 28x28 image (784 dimensions) and encode it
        # into a smaller representation of size latent_dim.
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim)
        )

        # --- Decoder ---
        # It will take the latent representation and reconstruct the
        # original 784-dimensional image.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Use Sigmoid to get pixel values between 0 and 1
        )

    def forward(self, x):
        # The forward pass chains the encoder and decoder.
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Instantiate the model and move it to the configured device
model = Autoencoder(LATENT_DIM).to(device)

# --- Loss Function and Optimizer ---
# We use Mean Squared Error loss, which measures the pixel-wise difference
# between the input and the reconstructed output.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)

# --- Training Loop ---
for epoch in range(EPOCHS):
    total_loss = 0
    for data in train_loader:
        img, _ = data  # We don't need the labels, just the images

        # Flatten the images from [batch_size, 1, 28, 28] to [batch_size, 784]
        img = img.view(img.size(0), -1).to(device)

        # --- Forward Pass ---
        output = model(img)
        loss = criterion(output, img)  # Compare reconstructed image with original

        # --- Backward Pass and Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # --- Print Statistics ---
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}')

print("Training finished!")


# --- Visualization ---
def visualize_reconstructions(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get a batch of test images
        data = next(iter(data_loader))
        imgs, _ = data
        imgs = imgs.to(device)

        # Get the reconstructions
        img_flat = imgs.view(imgs.size(0), -1)
        reconstructions = model(img_flat)

        # Reshape for plotting
        imgs = imgs.cpu().numpy()
        reconstructions = reconstructions.view(imgs.shape).cpu().numpy()

        # Plot the first 10 images and their reconstructions
        plt.figure(figsize=(20, 4))
        for i in range(10):
            # Display original
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(imgs[i].reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 4:
                ax.set_title("Original Images")

            # Display reconstruction
            ax = plt.subplot(2, 10, i + 1 + 10)
            plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 4:
                ax.set_title("Reconstructed Images")
        plt.show()


# Visualize the results on the test set
visualize_reconstructions(model, test_loader, device)
