import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import pickle

# Load the dataset
df = pd.read_csv("dataset.csv")
df = df.drop(["FlightNumber", "Flights","BoosterVersion", "Date", "LaunchSite"], axis=1)

df["GridFins"] = [int(x) for x in df["GridFins"]]
df["Reused"] = [int(x) for x in df["Reused"]]
df["Legs"] = [int(x) for x in df["Legs"]]

# Perform encoding
label_encoder = LabelEncoder()
for column in df.select_dtypes(include="object"):
    df[column] = label_encoder.fit_transform(df[column])


# Check for outliers
print(df.describe())

# Replace outliers
outliers = np.abs(df['PayloadMass'] - df["PayloadMass"].mean()) > 3 * df["PayloadMass"].std()
df.loc[outliers, "PayloadMass"] = df["PayloadMass"].mean()

outliers = np.abs(df['Serial'] - df["Serial"].mean()) > 3 * df["Serial"].std()
df.loc[outliers, "Serial"] = df["Serial"].mean()

# Split the dataset into features and target
X = df.drop("Class", axis = 1)
y = df["Class"]


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the SVM model
model = SVC()
model.fit(X_train, y_train)

# Set the hyperparameters
parameters = {
    "kernel": ["linear", "rbf"],
    "C": [1, 10, 100],
    "gamma": ["scale", "auto"],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(model, parameters, cv=5)

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Predict the test set
y_pred = model.predict(X_test)
print(y_pred)

# Calculate the accuracy
accuracy = np.mean(y_pred == y_test)

print("Accuracy:", accuracy)

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl','rb'))
