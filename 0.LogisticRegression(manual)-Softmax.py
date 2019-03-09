"""
    Pure Python
    Code by Tae Hwan Jung(@graykode)
"""
import numpy as np

x_data = [-2.0, -1.0, 1.0, 5.0]
y_data = [1.0, 1.0, 2.0, 3.0] # 3 classes

# Traninalbe Parameters
w = 0.0

def softmax(x):
    new_x_data = [w * i for i in x_data]
    return np.exp(w * x) / np.sum(np.exp(new_x_data))

# Loss function, Cross Entropy is loss function in Logistic Regression
# if you see Proof about Partial derivative of J(θ) see, https://deepnotes.io/softmax-crossentropy
def loss(x, y): # Cross Entropy function
    return  -(y * softmax(x) / len(x_data))

# compute gradient
def gradient(x, y):  # d_loss/d_w
    return softmax(x) - y / len(x_data)

# Training loop
for epoch in range(50):
    l = 0
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad # 0.01 is learning rate
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l += loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# test
for x in x_data:
    y_pred = softmax(x)
    print(round(y_pred, 5))