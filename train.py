import numpy as np

from data import load_data
from neural_network import NeuralNetwork

# load MNIST
x_train, y_train, x_test, y_test = load_data()

print("training images:", len(x_train))
print("testing images:", len(x_test))

#create network
network = NeuralNetwork()

#training settings

learning_rate = 0.01
epochs = 10

#training

for epoch in range(epochs):

    #shuffle training data
    indices = np.random.permutation(len(x_train))

    x_train = x_train[indices]
    y_train = y_train[indices]

    total_loss = 0
    correct = 0

    print(f"Epoch {epoch + 1}/{epochs}")

    for i in range(len(x_train)):

        #get one image and label

        x = x_train[i]
        y = y_train[i]

        #forward propogation

        prediction = network.forward(x)

        #calculate loss

        loss = network.loss(prediction, y)

        total_loss += loss

        #check prediction

        predicted_digit = np.argmax(prediction)

        if predicted_digit == y:
            correct += 1

        #backward propogation
        gradients = network.backward(x, y)

        #SGD update
        network.update(*gradients, learning_rate)

        #progress

        if i % 5000 == 0:

            accuracy = correct / (i + 1)

            print(f"image: {i:5d} | "
                  f"loss: {total_loss / (i + 1):.4f} | "
                  f"accuracy: {accuracy * 100:.4f}%"
                  )

    #epoch result 

    average_loss = total_loss / len(x_train)
    accuracy = correct / len(x_train)

    print("\nEpoch complete!")
    print(f"Average loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


#save trained model

np.savez(
    "trained_model.npz",
    W1 = network.W1,
    b1 = network.b1,
    W2 = network.W2,
    b2 = network.b2,
    W3 = network.W3,
    b3 = network.b3,
)

print("\nModel saved as trained_model.npz")