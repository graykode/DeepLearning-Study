"""
    Pure Python
    Code by Tae Hwan Jung(@graykode)
"""
# Answer y = x1*1 + x2*2 + x3*3
x_data = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]
y_data = [14.0, 32.0, 50.0]

# Traninalbe Parameters
w1 = 2.0
w2 = 2.0
w3 = 2.0

def forward(x1, x2, x3):
    return x1 * w1 + x2 * w2 + x3 * w3

# Loss function
def loss(x, y):
    y_pred = forward(x[0], x[1], x[2])
    return (y_pred - y) * (y_pred - y)

# compute gradient
def gradient1(x_vals, y):  # d_loss/d_w1
    return x_vals[0] * (x_vals[0] * w1 + x_vals[1] * w2 + x_vals[2] * w3 - y)

def gradient2(x_vals, y):  # d_loss/d_w2
    return x_vals[1] * (x_vals[0] * w1 + x_vals[1] * w2 + x_vals[2] * w3 - y)

def gradient3(x_vals, y):  # d_loss/d_w3
    return x_vals[2] * (x_vals[0] * w1 + x_vals[1] * w2 + x_vals[2] * w3 - y)

# Before training
print("Parameter w1 :%f, w2 : %f, w3 : %f" % (w1, w2, w3))

# Training loop
for epoch in range(100):
    l = 0
    for x_vals, y_val in zip(x_data, y_data):

        grad1 = gradient1(x_vals, y_val)  # d_loss/d_w1
        grad2 = gradient2(x_vals, y_val)  # # d_loss/d_w2
        grad3 = gradient3(x_vals, y_val)  # # d_loss/d_w3

        w1 = w1 - 0.01 * grad1
        w2 = w2 - 0.01 * grad2
        w3 = w3 - 0.01 * grad3

        print("\tgrad: ", x_vals, y_val, round(grad1, 2), round(grad2, 2), round(grad3, 2))
        l += loss(x_vals, y_val)

    print("progress:", epoch+1, "w1=%f, w2=%f, w3=%f"% (round(w1, 2), round(w2, 2), round(w3, 2)), "loss=", round(l, 2))

# After training
print("Parameter w1 :%f, w2 : %f, w3 : %f" % (w1, w2, w3))