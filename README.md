# PRACTICAL-3

# Lab No. 3: Implement a Feedforward Neural Network using TensorFlow or PyTorch

# Aim
To implement a Feedforward Neural Network (FNN) using PyTorch and TensorFlow (Keras) to perform binary classification on the AND logic gate dataset.

# Theory
A Feedforward Neural Network (FNN) is the simplest type of Artificial Neural Network in which information flows in only one direction—from the input layer to the output layer. There are no loops or feedback connections. The network learns by adjusting its weights using the backpropagation algorithm and an optimization algorithm such as Adam.
In this experiment, the AND gate dataset is used for binary classification. The network predicts whether the output should be 0 or 1 based on the two binary inputs.

# Algorithm
Import the required libraries.
Create the AND gate dataset.
Design a Feedforward Neural Network with input, hidden, and output layers.
Select Binary Cross-Entropy as the loss function.
Use the Adam optimizer.
Train the network for 100 epochs.
Predict the output for all input combinations.
Compare the predicted outputs with the expected outputs.

Program (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim

# Sample dataset: AND logic gate
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Define Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out

# Initialize model
model = FeedforwardNN()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Prediction
print("\nPredictions:")

with torch.no_grad():
    predictions = model(X)

    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].tolist()} => Predicted: {round(float(pred),4)} => Class: {int(pred>=0.5)}")

Output (PyTorch)
Predictions:

Input: [0.0, 0.0] => Predicted: 0.2082 => Class: 0
Input: [0.0, 1.0] => Predicted: 0.2371 => Class: 0
Input: [1.0, 0.0] => Predicted: 0.3124 => Class: 0
Input: [1.0, 1.0] => Predicted: 0.7344 => Class: 1

Program (TensorFlow/Keras)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample dataset: AND gate
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)

y = np.array([[0],
              [0],
              [0],
              [1]], dtype=np.float32)

# Build Feedforward Neural Network
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, y, epochs=100, verbose=1)

# Predictions
print("\nPredictions:")

predictions = model.predict(X)

for i in range(len(X)):
    print(f"Input: {X[i]} => Predicted: {predictions[i][0]:.4f} => Class: {int(predictions[i][0]>=0.5)}")



Expected Output (TensorFlow)
Predictions:

Input: [0. 0.] => Predicted: 0.08xx => Class: 0
Input: [0. 1.] => Predicted: 0.11xx => Class: 0
Input: [1. 0.] => Predicted: 0.13xx => Class: 0
Input: [1. 1.] => Predicted: 0.90xx => Class: 1
Note: The exact prediction values may vary slightly each time because the neural network starts with random weights.

Result
A Feedforward Neural Network was successfully implemented using both PyTorch and TensorFlow (Keras). The network was trained on the AND gate dataset and correctly classified all four input combinations after training.

Conclusion
The experiment demonstrated the implementation of a Feedforward Neural Network using two popular deep learning frameworks: PyTorch and TensorFlow. Both models successfully learned the AND logic gate through supervised learning. This experiment illustrates the basic working of neural networks, including forward propagation, loss calculation, backpropagation, and optimization, forming the foundation for solving more complex machine learning problems.

