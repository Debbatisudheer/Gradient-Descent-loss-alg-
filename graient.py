import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Initialize parameters
np.random.seed(0)
weights = np.random.randn(X_train_scaled.shape[1])
bias = 0


# Step 3: Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Step 6: Perform gradient descent
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    # Step 2: Compute predictions
    logits = np.dot(X_train_scaled, weights) + bias
    predictions = sigmoid(logits)

    # Step 4: Compute loss (Binary Cross-Entropy)
    loss = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))

    # Step 5: Compute gradients
    gradient_weights = np.dot(X_train_scaled.T, (predictions - y_train)) / len(X_train_scaled)
    gradient_bias = np.mean(predictions - y_train)

    # Step 6: Update parameters
    weights -= learning_rate * gradient_weights
    bias -= learning_rate * gradient_bias

    # Print loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")

# Step 7: Evaluate the model
logits_test = np.dot(X_test_scaled, weights) + bias
predictions_test = sigmoid(logits_test)
predicted_labels = np.round(predictions_test)

accuracy = np.mean(predicted_labels == y_test)
print("Accuracy on test set:", accuracy)