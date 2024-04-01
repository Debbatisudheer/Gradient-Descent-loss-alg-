Logistic Regression with Gradient Descent

This repository contains Python code for training a logistic regression model using gradient descent optimization. The model is trained on the diabetes dataset, and binary cross-entropy loss is used as the loss function.
Requirements

    Python 3.x
    numpy
    pandas
    scikit-learn
    matplotlib

Installation

    Clone this repository:

    bash

git clone https://github.com/your-username/logistic-regression-gradient-descent.git

Navigate to the project directory:

bash

cd logistic-regression-gradient-descent

Install the required dependencies:

bash

    pip install -r requirements.txt

Usage

    Ensure you have the dataset file named diabetes.csv in the project directory.
    Run the logistic_regression_gradient_descent.py script:

    bash

    python logistic_regression_gradient_descent.py

Description

    diabetes.csv: Dataset containing features and target variable (Outcome).
    logistic_regression_gradient_descent.py: Python script for training the logistic regression model using gradient descent.
    sigmoid: Sigmoid function used for logistic regression.
    binary_cross_entropy_loss: Binary cross-entropy loss function.
    gradient_descent: Function implementing gradient descent optimization to train the logistic regression model.
    The script visualizes the binary cross-entropy loss over iterations during training.

Output

mathematica

Binary Cross-Entropy Loss before Gradient Descent: 0.6931471805599453
Binary Cross-Entropy Loss after Gradient Descent: 0.5104918116017044

Steps of Gradient Descent

    Initialize Parameters: Start with initial values for the parameters (theta).
    Calculate Predictions: Compute the predicted probabilities using the sigmoid function.
    Calculate Loss: Compute the binary cross-entropy loss between the predicted probabilities and the actual labels.
    Calculate Gradients: Compute the gradients of the loss function with respect to each parameter.
    Update Parameters: Adjust the parameters in the direction that minimizes the loss, scaled by the learning rate.
    Repeat: Iterate steps 2-5 until convergence or a maximum number of iterations is reached.

    # Calculate predictions
y_pred = sigmoid(np.dot(X, theta))

# Calculate loss
loss = binary_cross_entropy_loss(y, y_pred)
losses.append(loss)

# Calculate gradients
gradient = np.dot(X.T, (y_pred - y)) / num_samples

# Update parameters
theta -= learning_rate * gradient

Interpretation of Loss Plot

    Convergence: Initially, the loss is relatively high, indicating poor performance of the model. As the number of iterations increases, the loss gradually decreases. This signifies that the optimization process is converging towards a minimum point.
    Decreasing Loss: The loss decreases over iterations, which demonstrates that the model is improving in its ability to make predictions. This reduction in loss indicates that the parameters (weights and bias) are being adjusted effectively to minimize the discrepancy between predicted and actual outcomes.
    Stability: The loss curve appears smooth, indicating stable optimization. This suggests that the chosen learning rate and other hyperparameters are appropriate for the problem, preventing oscillations or divergence during optimization.
    Final Loss: The final loss after Gradient Descent is lower than the initial loss, indicating that the model has been successfully trained and is performing better on the test data. This reduction in loss reflects the effectiveness of Gradient Descent in optimizing the parameters to improve model performance.

Loss Visualization

References

    Scikit-learn documentation
    NumPy documentation
    Pandas documentation
    Matplotlib documentation

Credits

This code is adapted from @sudheer debbati's implementation.
