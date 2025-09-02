import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# Generate the sine wave data
time_steps = np.linspace(0, np.pi * 10, 500, dtype=np.float32)
data = np.sin(time_steps)

# --- Create sequences and targets ---
# We'll use a sequence of 50 points to predict the 51st.
sequence_length = 50
X = []
y = []

for i in range(len(data) - sequence_length):
    # Input sequence
    X.append(data[i : i + sequence_length])
    # Target (the very next point)
    y.append(data[i + sequence_length])

combined = list(zip(X, y))

# 2. Shuffle the combined list
random.shuffle(combined)

# 3. Unzip the shuffled combined list back into two separate lists
X, y = zip(*combined)

# Convert to PyTorch tensors
X = torch.tensor(X).unsqueeze(-1) # Add a feature dimension
y = torch.tensor(y).unsqueeze(-1)

# --- Split data into training and test sets ---
# Let's use the first 400 sequences for training, the rest for testing
train_size = 400
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
for i in range(len(X)-train_size):
    for j in range(sequence_length):
        X_test[i][j] = 2*X_test[i][j]
    y_test[i] = 2*y_test[i]

print("X_train shape:", X_train.shape) # (num_samples, seq_length, num_features)
print("y_train shape:", y_train.shape)   # (num_samples, num_features)


class SineRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=70, output_size=1, num_layers=1):
        super(SineRNN, self).__init__()
        self.hidden_size = hidden_size

        # Define the RNN layer
        # batch_first=True makes the input/output shape (batch, seq, feature)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layer that maps from hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Initialize hidden state with zeros.
        # h0 shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Pass data through RNN layer
        # rnn_out contains the output for each time step
        # hidden contains the final hidden state
        rnn_out, hidden = self.rnn(x, h0)

        # We only want the output from the LAST time step.
        # rnn_out[:, -1, :] gives us the hidden state of the last element in the sequence
        last_time_step_out = rnn_out[:, -1, :]

        # Pass the last output through the linear layer
        out = self.fc(last_time_step_out)

        return out

if __name__ == '__main__':
    # Instantiate the model, define loss and optimizer
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = SineRNN().to(device)

    # --- FIX IS HERE ---
    # Move your training and test data to the selected device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)  # Good practice to move the test set too
    y_test = y_test.to(device)  # Good practice to move the test set too
    # -------------------

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 200

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Set model to evaluation mode
    model.eval()

    # We'll generate predictions starting from the last sequence in the training data
    test_input = X_train[-1].unsqueeze(0)  # Get the last sequence and add a batch dimension

    predictions = []

    with torch.no_grad():
        for _ in range(len(X_test)):
            # Get the prediction for the next point
            pred = model(test_input)
            predictions.append(pred.item())

            # Update the input sequence:
            # Remove the first element and append the prediction
            new_sequence = test_input.squeeze(0)[1:].tolist()
            new_sequence.append([pred.item()])
            test_input = torch.tensor(new_sequence).unsqueeze(0).to(device)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.title('Sine Wave Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)

    # --- Formal Evaluation on the Test Set ---

    # Set the model to evaluation mode
    model.eval()

    # We don't need to calculate gradients for evaluation, so we use torch.no_grad()
    with torch.no_grad():
        # 1. Get predictions for the entire test set at once
        test_predictions = model(X_test)

        # 2. Compare the predictions with the actual target values
        test_loss = criterion(test_predictions, y_test)

    print(f"\nMean Squared Error on Test Set: {test_loss.item():.4f}")

    # You can also plot these predictions against the true values
    plt.figure(figsize=(12, 6))
    plt.title("Model Predictions vs. Actual Data on Test Set")
    # We use y_test.numpy() to get the true values for plotting
    plt.plot(y_test.cpu().numpy(), label='Actual Data')
    # We use test_predictions.numpy() for the model's output
    plt.plot(test_predictions.cpu().numpy(), label='Model Predictions', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()
