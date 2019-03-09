"""
    Pure Python
    Code by Tae Hwan Jung(@graykode)
"""
import numpy as np

x_data = [-2.0, -1.0, 1.0, 5.0]
y_data = [1.0, 1.0, 2.0, 2.0]

# Traninalbe Parameters
w = -10.0

def sigmoid(x):
    return 1 / (1 + np.exp(-w * x))

# Loss function
def loss(x, y):
    return  -(y * np.log(sigmoid(x)) + (1-y) * np.log(1- sigmoid(x)))

# compute gradient
# if you see Proof about Partial derivative of J(θ) see, https://wikidocs.net/4289
def gradient(x, y):  # d_loss/d_w
    return x * (sigmoid(x) - y)

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
    y_pred = sigmoid(x)
    print(round(y_pred, 5))