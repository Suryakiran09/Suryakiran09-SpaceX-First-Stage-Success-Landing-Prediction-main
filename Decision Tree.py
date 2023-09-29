import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("dataset_part_2.csv")
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

# Create the Decision Tree Regression model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")