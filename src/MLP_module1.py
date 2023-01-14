import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv("C:\\Users\\Jianw\\Documents\\GitHub\\aiap13-liaw-jian-wei-026z\\data aiap13\\failure.csv")

# Perform preprocessing
data.dropna(inplace=True)
data = pd.get_dummies(data)

# Define the features and target
X = data.drop("Failure", axis=1)
y = data["Failure"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a dictionary to store the models
models = {"Random Forest": RandomForestClassifier(),
          "SVM": SVC(),
          "Logistic Regression": LogisticRegression()}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} - accuracy: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")
