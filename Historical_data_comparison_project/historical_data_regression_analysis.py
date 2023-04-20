import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob # for directory parsing
from openpyxl import load_workbook  # attempt 2 at fast loading excel worksheets
from datetime import date
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats

##### DEFINED PARAMETERS
attr = 'liver_volume'

# Define paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
input_dir = os.path.join(parent_dir, "Data", "Historical_01", "Input")
output_dir = os.path.join(parent_dir, "Data", "Historical_01", "Output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Import data
df = pd.read_csv(os.path.join(input_dir, "Abdomen CT parameters vs Patient age, sex - out.csv"))
df = df[[attr, 'patient_age']]

# Exclude outliers in the dataset
Q1 = df[[attr, 'patient_age']].quantile(0.25)
Q3 = df[[attr, 'patient_age']].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df_filtered = df[(df[[attr, 'patient_age']] >= lower_bound) & (df[[attr, 'patient_age']] <= upper_bound)].dropna()
X = df_filtered[['patient_age']]
y = df_filtered[attr]

# Divide the data into training and testing datasets (70-30)
#rs = 42 # random state parameter for reproducibility
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X, y)

# Make predictions on the data
y_pred = model.predict(X)

# Create a figure to test 3 assumptions (Linearity, Homoscedasticity, and Normality)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Scatter plot of actual vs predicted values (Linearity Test)
axes[0].scatter(y_pred, y)
min_val = 1.05*(min(min(y_pred), min(y)))
max_val = 1.05*(max(max(y_pred), max(y)))
axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--') # Adding a straight diagonal line
axes[0].set_xlim(min_val, max_val)
axes[0].set_ylim(min_val, max_val)
axes[0].set_ylabel(f"Actual {attr}")
axes[0].set_xlabel(f"Predicted {attr}")
axes[0].set_title("Linearity Test: Actual vs Predicted plot")
# Residual plot (Homoscedasticity Test)
residuals = y - y_pred
axes[1].scatter(y_pred, residuals)
axes[1].axhline(y=0, color='r', linestyle='-')
axes[1].set_xlim(min_val, max_val)
axes[1].set_xlabel(f"Predicted {attr}")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Homoscedasticity Test: Residual Plot")
# Q-Q plot (Normality Test)
sm.qqplot(residuals, line='45', fit=True, ax=axes[2])
axes[2].set_title("Normality Test: Q-Q Residuals Plot")
# Adjust the layout and display the figure
plt.tight_layout()
plt.show()
# Shapiro-Wilk test for normality
W, p = stats.shapiro(residuals)
print("Shapiro-Wilk Normality test: W =", W, ", p-value =", p)

# Calculate the evaluation metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("The Mean Squared Error represents the average squared difference between the actual and predicted values.")
print("Root Mean Squared Error (RMSE):", rmse)
print("The Root Mean Squared Error is the square root of the MSE, and it represents the average difference between the actual and predicted values.")
print("Mean Absolute Error (MAE):", mae)
print("The Mean Absolute Error represents the average absolute difference between the actual and predicted values.")
print("R-squared (R2):", r2)
print("The R-squared value represents the proportion of the variance in the actual values that is predictable from the independent variable(s). It ranges from 0 to 1.")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Actual data', alpha=0.7)
plt.plot(X, y_pred, color='red', label='Line of best fit (Least Squares Regression)')
plt.ylim(min_val, max_val)
plt.xlabel("Patient Age (years)")
plt.ylabel(f"{attr}")
plt.title(f"Linear Regression of {attr} vs Patient Age")
plt.legend()

# Display the model's coefficients
coef_interpretation = f"y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * age"
plt.text(0.05, 0.95, coef_interpretation, transform=plt.gca().transAxes, fontsize=12)
plt.show()

# Interpret the model
print("Intercept (b0):", model.intercept_)
print("Coefficient for Age (b1):", model.coef_[0])
print("\nInterpretation:")
print(f"For every one-year increase in patient age, the {attr} is expected to change by {model.coef_[0]:.2f} units, assuming a linear relationship between age and {attr}.")
print(f"The intercept represents the estimated {attr} then the patient's age is 0.")



#
f = 1