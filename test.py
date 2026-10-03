import numpy as np

from data import load_data
from neural_network import NeuralNetwork

# load MNIST
x_train, y_train, x_test, y_test = load_data()

#create network
network = NeuralNetwork()

# Load trained weights

model = np.load("trained_model.npz")

network.W1 = model["W1"]
network.b1 = model["b1"]

network.W2 = model["W2"]
network.b2 = model["b2"]

network.W3 = model["W3"]
network.b3 = model["b3"]


print("Trained model loaded!")

#test

correct = 0

for i in range (len(x_test)):

    x = x_test[i]
    y = y_test[i]

    prediction = network.forward(x)

    predicted_digit = np.argmax(prediction)

    if predicted_digit == y:
        correct += 1

accuracy = correct / len(x_test)

print("\n------TEST RESULTS------")
print("Test images:", len(x_test))
print(f"Correct: {correct}")
print(f"Wrong: {len(x_test) - correct}")
print(f"Test accuracy: {accuracy * 100:.2f}%")