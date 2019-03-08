"""
    Pure Python
    Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/02_manual_gradient.py
"""
# Answer y = x * 3
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 6.0, 9.0]

# Traninalbe Parameters
w = 1.0

def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)

# Before training
print("Parameter w : %f, Predict y : %f" % (w, forward(4)))

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad # 0.01 is learning rate
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Parameter w : %f, Predict y : %f" % (w, forward(4)))