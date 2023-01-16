import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("C:\\Users\\Jianw\\Documents\\GitHub\\aiap13-liaw-jian-wei-026z\\data aiap13\\failure.csv")

# Perform preprocessing
data.dropna(inplace=True)
data = pd.get_dummies(data)

# Define the features and target
X = data.drop("Failure", axis=1)
y = data["Failure"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

# Train and evaluate a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy for Logistic Regression:", accuracy_score(y_test, y_pred))
print("Confusion Matrix for Logistic Regression:\n", confusion_matrix(y_test, y_pred))

# Train and evaluate a decision tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Accuracy for Decision Tree:", accuracy_score(y_test, y_pred))
print("Confusion Matrix for Decision Tree:\n", confusion_matrix(y_test, y_pred))

# Train and evaluate a random forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy for Random Forest:", accuracy_score(y_test, y_pred))
print("Confusion Matrix for Random Forest:\n", confusion_matrix(y_test, y_pred))

# Visualize the results
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_score(y_test, log_reg.predict(X_test)), accuracy_score(y_test, dt.predict(X_test)), accuracy_score(y_test, rf.predict(X_test))]
plt.bar(models, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()