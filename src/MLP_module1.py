import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("failures.csv")

# Perform preprocessing
data.dropna(inplace=True)
data = pd.get_dummies(data)

# Perform feature engineering
data["new_feature"] = data["feature1"] + data["feature2"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
