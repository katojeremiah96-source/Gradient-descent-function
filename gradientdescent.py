import numpy as np
 
# 1. Training data
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3]
], dtype=float)
 
# Expected outputs
y = np.array([8, 9, 18, 19], dtype=float).reshape(-1, 1)
 
# 2. Initialize weights and bias
np.random.seed(42)
weights = np.zeros((2, 1))
bias = 0.0
 
# 3. Set learning rate and number of training iterations
learning_rate = 0.01
epochs = 1000
 
# 4. Train using Gradient Descent
for epoch in range(epochs):
 
    # Forward propagation
    predictions = X @ weights + bias
 
    # Calculate Mean Squared Error
    error = predictions - y
    loss = np.mean(error ** 2)
 
    # Calculate gradients (partial derivatives)
    dw = (2 / len(X)) * (X.T @ error)
    db = 2 * np.mean(error)
 
    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db
 
    # Display training progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
 
# 5. Display final parameters
print("\nTraining completed!")
print("Updated weights:", weights.flatten())
print("Updated bias:", bias)
 
# 6. Test the trained model
test_data = np.array([[5, 2]])
prediction = test_data @ weights + bias
print("Predicted output:", prediction[0, 0])