# Lab No. 3: Implement a Feedforward Neural Network using PyTorch
# Aim

To implement a Feedforward Neural Network (FNN) using PyTorch and classify the AND logic gate dataset.

# Theory

A Feedforward Neural Network (FNN) is the simplest type of Artificial Neural Network in which information flows only in one direction—from the input layer to the hidden layer and finally to the output layer. It does not contain any feedback or recurrent connections.

In this experiment, the network consists of:

An input layer with 2 neurons.
A hidden layer with 4 neurons using the ReLU activation function.
An output layer with 1 neuron using the Sigmoid activation function.

The model is trained using the Binary Cross-Entropy (BCELoss) loss function and the Adam optimizer to classify the AND logic gate.

# Algorithm
Import the required PyTorch libraries.
Create the AND gate dataset.
Define the Feedforward Neural Network architecture.
Initialize the model, loss function, and optimizer.
Train the network for 100 epochs.
Calculate the loss and update the weights using backpropagation.
Predict the outputs after training.
Display the predicted class for each input.
Program

(Paste the PyTorch code exactly as provided by your instructor.)

Output
Predictions:

Input: [0.0, 0.0] => Predicted: 0.2082 => Class: 0
Input: [0.0, 1.0] => Predicted: 0.2371 => Class: 0
Input: [1.0, 0.0] => Predicted: 0.3124 => Class: 0
Input: [1.0, 1.0] => Predicted: 0.7344 => Class: 1

Note: The prediction values may vary slightly each time because the neural network starts with random weights.

Result

The Feedforward Neural Network was successfully implemented using PyTorch. The model correctly learned the AND gate and classified all four input combinations.

Conclusion

The experiment demonstrated the implementation of a Feedforward Neural Network using PyTorch. The network successfully learned the AND logic gate through forward propagation, backpropagation, and weight optimization using the Adam optimizer. This experiment provides a basic understanding of neural networks used in binary classification tasks.
