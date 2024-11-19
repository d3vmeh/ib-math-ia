import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
from PIL import Image 
import random


def load_weights():
    global W1, b1, W2, b2
    weights = np.load('weights.npz')
    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']


    loaded = pickle.load(open("weights.pkl", "rb"))

    W1 = loaded[0]
    b1 = loaded[1]
    W2 = loaded[2]
    b2 = loaded[3]

    return W1, b1, W2, b2



# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0  # Flatten the images and normalize
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

# One-hot encoding for labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Initialize parameters
input_size = 784  # 28x28 pixels
hidden_size = 128  # Number of neurons in hidden layer
output_size = 10   # Number of classes (digits 0-9)

np.random.seed(42)
# W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights between input and hidden layer
# b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
# W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights between hidden and output layer
# b2 = np.zeros((1, output_size))  # Bias for output layer

load_weights()
# Activation and loss functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward pass
def forward_pass(X):
    Z1 = np.dot(X, W1) + b1  # Input to hidden layer
    A1 = sigmoid(Z1)  # Activation for hidden layer
    Z2 = np.dot(A1, W2) + b2  # Input to output layer
    A2 = softmax(Z2)  # Output layer predictions
    return Z1, A1, Z2, A2

# Backward pass (gradient descent)
def backward_pass(X, y_true, Z1, A1, Z2, A2):
    global W1, b1, W2, b2
    m = X.shape[0]  # Number of samples
    
    # Output layer error
    dZ2 = A2 - y_true  # Derivative of loss w.r.t Z2
    dW2 = np.dot(A1.T, dZ2) / m  # Derivative of loss w.r.t W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Derivative of loss w.r.t b2
    
    # Hidden layer error
    dA1 = np.dot(dZ2, W2.T)  # Backpropagate error to hidden layer
    dZ1 = dA1 * sigmoid_derivative(A1)  # Derivative of loss w.r.t Z1
    dW1 = np.dot(X.T, dZ1) / m  # Derivative of loss w.r.t W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Derivative of loss w.r.t b1
    
    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Training the model
def train(X_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_pass(X_train)  # Forward pass
        loss = cross_entropy_loss(y_train, A2)  # Calculate loss
        backward_pass(X_train, y_train, Z1, A1, Z2, A2)  # Backward pass (gradient descent)
        
        #if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Hyperparameters
epochs = 1000
learning_rate = 0.1

print(f"{len(X_train)} training samples")
# Train the model
#train(X_train, y_train, epochs, learning_rate)


#print("Final weights:", W1, b1, W2, b2)
# Prediction function
def predict(X):
    _, _, _, A2 = forward_pass(X)
    return np.argmax(A2, axis=1)
ಠ_ಠ = 0

def save_weights():
    np.savez('weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)
    pickle.dump([W1,b1,W2,b2], open("ib-math-ia/weights.pkl", "wb"))


#save_weights()


W1, b1, W2, b2 = load_weights()

#print("Here are the weights", W1, b1, W2, b2)
# Evaluate the model on test data
y_pred = predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_test_labels)

print(predict(X_test[0:10]))
print(y_test[0:10])

print(f'Test accuracy: {accuracy}')

def test_prediction(index):
    current_image = X_test[None,index,:]

    prediction = predict(X_test[None,index,:])
    
    #img = img.resize((1, 784))  # Resize to 28x28 pixels
    #img = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0  # Normalize and reshape
    #img = np.array(img).reshape(1, 784).astype('float32') / 255.0

    prediction = predict(current_image)
    label = y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", np.argmax(label))
    print(current_image.shape)
    #current_image = current_image.reshape((28, 28)) * 255
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


index = random.randint(1,1000)
test_prediction(index)


