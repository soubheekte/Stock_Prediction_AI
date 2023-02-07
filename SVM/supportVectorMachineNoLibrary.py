import numpy as np

# Define the training data
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y = np.array([1, 1, -1, -1])

# Define the parameters of the model
learning_rate = 0.1
num_iterations = 100

# Initialize the weight and bias
w = np.zeros(X.shape[1])
b = 0

# Define the hinge loss function
def hinge_loss(w, b, X, y):
    loss = 0
    for i in range(X.shape[0]):
        loss += max(0, 1 - y[i] * (np.dot(w, X[i]) + b))
    return loss

# Train the model
for i in range(num_iterations):
    loss = hinge_loss(w, b, X, y)
    gradient_w = np.zeros(w.shape)
    gradient_b = 0
    for j in range(X.shape[0]):
        if y[j] * (np.dot(X[j], w) + b) < 1:
            gradient_w += -y[j] * X[j]
            gradient_b += -y[j]
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

# Print the final weight and bias
print("Weight:", w)
print("Bias:", b)



# In this example, we first define the training data and the parameters of the model. 
# We then initialize the weight and bias to zero. We define a hinge loss function which calculates the loss 
# for each iteration. Then we train the model by calculating the gradient of weight and bias and updating the 
# weight and bias by subtracting the product of learning rate and gradient from weight and bias respectively. 
# Finally, we print the final weight and bias.

# You can also test this model on new data points by using the following code snippet:

# Define new data point
test_point = np.array([2, 2])

# Make prediction
prediction = np.sign(np.dot(test_point, w) + b)
print("Prediction: ", prediction)


# Please keep in mind that this is a very simple example with a small dataset, 
# and in practice you would need to use more sophisticated algorithms, such as the 
# Sequential Minimal Optimization (SMO) algorithm, to train an SVM on large datasets.