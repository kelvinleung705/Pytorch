import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# Generate the sine wave data
time_steps = np.linspace(0, np.pi * 20, 500, dtype=np.float32)
data_yes = np.sin(time_steps)
time_steps = np.linspace(0, np.pi * 20, 500, dtype=np.float32)
data_no = np.tan(time_steps)
data_no - np.clip(data_no, -10, 10)

# --- Create sequences and targets ---
# We'll use a sequence of 50 points to predict the 51st.
sequence_length = 50
X = []
y = []

for i in range(len(data_yes) - sequence_length):
    # Input sequence
    X.append(data_yes[i : i + sequence_length])
    # Target (the very next point)
    y.append(1)
    # Input sequence
    X.append(data_no[i: i + sequence_length])
    # Target (the very next point)
    y.append(0)

combined = list(zip(X, y))

# 2. Shuffle the combined list
random.shuffle(combined)

# 3. Unzip the shuffled combined list back into two separate lists
X, y = zip(*combined)

# Convert to PyTorch tensors
X = torch.tensor(X).unsqueeze(-1) # Add a feature dimension
y = torch.tensor(y).unsqueeze(-1).float()

# --- Split data into training and test sets ---
# Let's use the first 400 sequences for training, the rest for testing
train_size = 800
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
for i in range(len(X)-train_size):
    for j in range(sequence_length):
        X_test[i][j] = X_test[i][j]
    y_test[i] = y_test[i]

print("X_train shape:", X_train.shape) # (num_samples, seq_length, num_features)
print("y_train shape:", y_train.shape)   # (num_samples, num_features)


class SineLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=1, num_layers=1):
        super(SineLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Define the RNN layer
        # batch_first=True makes the input/output shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layer that maps from hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        # h0 shape: (num_layers, batch_size, hidden_size)
        # c0 shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_size).to(x.device)

        # We pass the input and the initial hidden/cell states to the LSTM
        # The LSTM returns the output for each time step, plus the final hidden and cell states
        out, _ = self.lstm(x, (h0, c0))

        # We are only interested in the output of the *last* time step.
        # out shape: (batch_size, seq_length, hidden_size)
        # out[:, -1, :] selects the last time step for all batches
        last_time_step_out = out[:, -1, :]

        # Pass the last time step's output to the fully-connected layer
        final_out = self.fc(last_time_step_out)
        return final_out

if __name__ == '__main__':
    # Instantiate the model, define loss and optimizer
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = SineLSTM().to(device)

    # --- FIX IS HERE ---
    # Move your training and test data to the selected device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)  # Good practice to move the test set too
    y_test = y_test.to(device)  # Good practice to move the test set too
    # -------------------

    criterion = nn.BCEWithLogitsLoss()  # Mean Squared Error for regression
    learning_rate = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 400

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

    print("\n--- Plotting Model Confidence ---")

    # We already have these from the accuracy calculation, but let's show them again
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_probs = torch.sigmoid(test_logits)  # Probabilities are on the GPU
        predicted_classes = (test_probs > 0.5).float()

    # Move all necessary data to the CPU and NumPy for plotting
    y_test_cpu = y_test.cpu().numpy().flatten()
    predicted_classes_cpu = predicted_classes.cpu().numpy().flatten()
    test_probs_cpu = test_probs.cpu().numpy().flatten()
    num_test_samples = len(y_test_cpu)

    # Separate the points for color-coding the plot
    correct_sine_probs = []
    correct_sine_indices = []
    correct_square_probs = []
    correct_square_indices = []
    incorrect_probs = []
    incorrect_indices = []

    for i in range(num_test_samples):
        is_correct = (predicted_classes_cpu[i] == y_test_cpu[i])
        is_sine = (y_test_cpu[i] == 1)

        if is_correct:
            if is_sine:
                correct_sine_probs.append(test_probs_cpu[i])
                correct_sine_indices.append(i)
            else:  # Is a square wave
                correct_square_probs.append(test_probs_cpu[i])
                correct_square_indices.append(i)
        else:  # Incorrect prediction
            incorrect_probs.append(test_probs_cpu[i])
            incorrect_indices.append(i)

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot the points
    plt.scatter(correct_sine_indices, correct_sine_probs, color='green', label='Correctly Identified as Sine',
                alpha=0.7, marker='o')
    plt.scatter(correct_square_indices, correct_square_probs, color='blue', label='Correctly Identified as Not Sine',
                alpha=0.7, marker='o')
    plt.scatter(incorrect_indices, incorrect_probs, color='red', label='Incorrect Prediction', alpha=1.0, marker='x',
                s=100)  # 's' makes the 'x' bigger

    # Add the decision boundary line
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary (0.5)')

    test_predictions = model(X_test)

    # 2. Compare the predictions with the actual target values
    test_loss = criterion(test_predictions, y_test)
    print(f"\nMean Squared Error on Test Set: {test_loss.item():.4f}")
    # Add labels and title
    plt.title('Model Confidence on Test Set')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Predicted Probability of Being a Sine Wave')
    plt.ylim(-0.1, 1.1)  # Set y-axis limits
    plt.legend()
    plt.grid(True)
    plt.show()
