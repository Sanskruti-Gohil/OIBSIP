# Task - 5
# Predicting House prices using Linear Regression

import pandas as pd

# Loading the dataset
df = pd.read_csv("C:\\Users\\spgoh\\Downloads\\archive (1)\\1553768847-housing.csv")

df.head()
df.shape
df.info()

df.describe()

df.isnull().sum()

# Handle missing values (numeric columns only)
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# ------------------------------------
# IMPORTANT: Dataset-specific fixes
# ------------------------------------

# Remove this line (columns do NOT exist in this dataset)
# df.drop(['id', 'address'], axis=1, inplace=True)

# Encode categorical column
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Feature selection
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Visualization
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Model coefficients
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coeff_df)
