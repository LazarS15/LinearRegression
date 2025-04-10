import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegression

# Test on 1 feature data
X_single = np.array([[1], [2], [3], [4]])
y_single = np.array([2, 5, 9, 11])

model_single = LinearRegression(learning_rate=0.1, n_iter=700)
model_single.fit(X_single, y_single)

print("Single feature results:")
print(f"Weights: {model_single.w}")
print(f"Bias: {model_single.b:.2f}")

# Visualization for 2d data
x_line = np.linspace(0, 5, 100).reshape(-1, 1)
y_line = model_single.predict(x_line)
plt.scatter(X_single, y_single, color='blue', label='Original data')
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.title("Linear Regression (1 Feature)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Test on multiple feature data (3 features)
X_multi = np.array([
    [1, 2, 1],
    [2, 3, 0],
    [3, 4, 2],
    [4, 5, 1],
])
y_multi = np.array([7, 13, 16, 22])

model_multi = LinearRegression(learning_rate=0.01, n_iter=5000)
model_multi.fit(X_multi, y_multi)

print("Multiple feature results:")
print(f"Weights: {model_multi.w}")
print(f"Bias: {model_multi.b:.2f}")