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


    # Gradient Descent
def gradient_descent(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)  # Initialize weights (coefficients) to zeros
    losses = []

    for _ in range(num_iterations):
        # Calculate predictions
        y_pred = sigmoid(np.dot(X, theta))  # Compute predictions using current weights

        # Calculate loss
        loss = binary_cross_entropy_loss(y, y_pred)  # Compute loss
        losses.append(loss)

        # Calculate gradients
        gradient = np.dot(X.T, (y_pred - y)) / num_samples  # Compute gradients

        # Update parameters (weights and bias)
        theta -= learning_rate * gradient  # Adjust weights (coefficients)

    return theta, losses

    In this function:

    The theta variable represents the parameters of the logistic regression model, which are the weights (coefficients). It is initialized with zeros.
    Inside the loop, the predictions (y_pred) are computed using the current set of weights (theta) by performing a dot product between the feature matrix X and the weights.
    The loss is computed using the binary cross-entropy loss function.
    Gradients are calculated with respect to the loss function using the current set of predictions and the actual target values.
    Finally, the parameters (weights) are updated by subtracting a fraction of the gradients, scaled by the learning rate (learning_rate).

So, the adjustment of parameters (weights) occurs during the gradient descent process within the gradient_descent function.



When working with gradient descent and loss functions for diabetes prediction, you should consider the following measures:

    Choose appropriate features: Select relevant features (such as glucose levels, BMI, blood pressure, etc.) that are likely to be predictive of diabetes. Ensure these features adequately represent the variability present in the data.

    Preprocess the data: Clean the data by handling missing values, normalizing or standardizing the features, and encoding categorical variables if necessary. This ensures that the data is suitable for training the model.

    Split the data: Divide the dataset into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance. This helps assess how well the model generalizes to unseen data.

    Define the model architecture: Choose an appropriate model architecture for diabetes prediction, such as logistic regression, decision trees, or neural networks. Define the structure of the model, including the number of layers, neurons, and activation functions.

    Choose a loss function: Select a suitable loss function for the task of diabetes prediction. Common choices include binary cross-entropy loss for binary classification tasks like diabetes prediction.

    Select a learning rate: Experiment with different learning rates to find one that allows the model to converge effectively without overshooting the optimal solution. A learning rate that is too high may cause the model to diverge, while a learning rate that is too low may result in slow convergence.

    Implement gradient descent: Use gradient descent optimization to train the model. Iterate through the training data, compute the gradients of the loss function with respect to the model parameters, and update the parameters accordingly to minimize the loss.

    Monitor convergence: Keep track of the loss function's value during training. Ensure that it decreases over time and converges to a stable value. If the loss function does not converge or exhibits erratic behavior, adjust the learning rate or model architecture accordingly.

    Evaluate the model: Once training is complete, evaluate the model's performance on the testing set. Calculate relevant metrics such as accuracy, precision, recall, and F1-score to assess how well the model predicts diabetes.

    Iterate and refine: Based on the evaluation results, iterate and refine the model by adjusting hyperparameters, trying different model architectures, or incorporating additional features. Continuously refine the model until satisfactory performance is achieved.

By following these measures, you can effectively utilize gradient descent and loss functions for diabetes prediction, ultimately developing a model that accurately identifies individuals at risk of diabetes.

Credits

This code is adapted from @sudheer debbati's implementation.
