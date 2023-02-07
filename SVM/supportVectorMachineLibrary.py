from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a SVM model with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the model on the training data
clf.fit(X_train, y_train)

# Test the model on the test data
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)




# In this example, we first load the Iris dataset using the scikit-learn datasets module. We then split the data into training and test sets using the train_test_split function. Next, we create an instance of the SVC class with a linear kernel and train the model on the training data using the fit method. Finally, we test the model on the test data and print the accuracy.

# You can also use different types of kernel functions, like 'rbf', 'poly', etc..

# python
# Copy code
# clf = svm.SVC(kernel='rbf')
# This code uses the scikit-learn library to train a SVM model using the Iris dataset. You can apply this model to other datasets and adjust the parameters of the model according to the nature of the problem.


