import numpy as np

X = np.array([[1, 3], [4, 10]], dtype=float)
y = np.array([[5], [6]], dtype=float)

m = np.array([[-1], [2]], dtype=float)
b = 1.0

alpha = 0.001
N_ITER = 3

print("Initial parameters:")
print(f"m = {m.ravel()}")
print(f"b = {b}")
print()

for i in range(N_ITER):
    y_pred = X @ m + b
    error = y_pred - y

    m_grad = (2 / X.shape[0]) * X.T @ error
    b_grad = (2 / X.shape[0]) * np.sum(error)

    m = m - alpha * m_grad
    b = b - alpha * b_grad

    print(f"Iteration {i + 1}:")
    print(f"  y_pred = {y_pred.ravel()}")
    print(f"  error  = {error.ravel()}")
    print(f"  m_grad = {m_grad.ravel()}")
    print(f"  b_grad = {b_grad:.4f}")
    print(f"  m      = {m.ravel()}")
    print(f"  b      = {b:.4f}")
    print()
