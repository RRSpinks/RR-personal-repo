import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import diagnostic as sms_diag
from statsmodels.stats import stattools as sms
from itertools import zip_longest as lzip
from statsmodels.stats.api import linear_harvey_collier as sms_linear_hc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.random as rnd


########################################## INTRO #############################################
# This script seeks to apply a basic linear regression model and assess if the data is suitable for this type of regression analysis


#### 
#### First we must load the data, specifying the type of dataset
# Choose the dataset
type = 0 # Choose the type of data:
            # 0 = User
            # 1 = Real life example
            # 2 = Synthetic example
            # 3 = Synthetic example, normality issue
            # 4 = Synthetic example, homoscedasticity issue
            # 5 = Synthetic example, linearity issue

rnd.seed(5) # specify seed for reproducibility
if type == 0:
    # User can specify data here
    data = pd.read_csv('/home/richard/Richard/RR-personal-repo/Data/Historical_01/Input/Abdomen CT parameters vs Patient age, sex - out.csv') # User path here
    X = data['patient_age']
    y = data['liver_volume']
    # Remove outliers
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((y < (Q1 - 1.5 * IQR)) | (y > (Q3 + 1.5 * IQR)))
    data_clean = data[~outliers]
    X = data_clean['patient_age']
    y = data_clean['liver_volume']
    X = sm.add_constant(X)

if type == 1:
    # Example dataset 1  (This is real life data that satisfies all assumptions for linear regression)
    data = sm.datasets.get_rdataset("longley", "datasets").data
    X = data[['GNP.deflator']]
    y = data['Employed']
    X = sm.add_constant(X)

if type == 2:
    # Example dataset 2  (This is synthetic data that satisfies all assumptions for linear regression)
    x = np.linspace(0, 10, 100)
    y = 3 * x + 5 + rnd.normal(0, 0.5, 100)  # some added noise
    data = pd.DataFrame({'X': x, 'Y': y})
    X = data[['X']]
    y = data['Y']
    X = sm.add_constant(X)

if type == 3:
    # Example dataset 3  (This is synthetic data that violates the normality assumptions of linear regression)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 3 + rnd.lognormal(0, 0.5, 100)  # Linear relationship with log-normal noise
    data = pd.DataFrame({'X': x, 'Y': y})
    X = data[['X']]
    y = data['Y']
    X = sm.add_constant(X)

if type == 4:
    # Example dataset 4  (This is synthetic data that violates the homoscedasticity assumptions of linear regression)
    x = np.linspace(0, 10, 100)
    y = 2 * x + rnd.normal(0, x, 100)  # Linear relationship with increasing (heteroscedastic) noise, note: will also be non-normal distribution
    data = pd.DataFrame({'X': x, 'Y': y})
    X = data[['X']]
    y = data['Y']
    X = sm.add_constant(X)

if type == 5:
    # Example dataset 5  (This is synthetic data that violates the homoscedasticity assumptions of linear regression)
    x = np.linspace(0, 10, 100)
    y = x**2 + rnd.normal(0, 1, 100)  # Quadratic relationship, note: will also be non-normal distribution for linear, but would be normal if it was a non-linear model
    data = pd.DataFrame({'X': x, 'Y': y})
    X = data[['X']]
    y = data['Y']
    X = sm.add_constant(X)

#### 
#### Apply an OLS linear regression model to the data and print the model summary
# Fit the linear regression model
model = sm.OLS(y, X).fit()
print(model.summary())

####
#### Now we want to generate a scatterplot of the data fit with the OLS model. We also want to generate diagnostics plots to accompany the main plot
# Create the figure with the specified layout
fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(3, 2, height_ratios=[2.2, 1, 1], width_ratios=[1, 1])

# Scatterplot of original data with the linear regression model
x_values = np.linspace(min(X.iloc[:, 1]), max(X.iloc[:, 1]), 100) # Calculate the line of best fit
y_values = model.params[0] + model.params[1] * x_values
ax0 = plt.subplot(gs[0, :])
ax0.scatter(X.iloc[:, 1], y, alpha=0.7, label="Data")
ax0.plot(x_values, y_values, color='red', label="OLS Line of Best Fit")
ax0.set_xlabel(X.columns[1])
ax0.set_ylabel(y.name)
ax0.set_title("Scatterplot of Original Data with Linear Regression Model")
ax0.legend()

# Residuals vs. fitted values plot
ax1 = plt.subplot(gs[1, 0])
ax1.axhline(y=0, color='red', linestyle='-')
ax1.scatter(model.fittedvalues, model.resid, alpha=0.7)
max_residual = max(abs(model.resid))
ax1.set_ylim(-1.1*max_residual, 1.1*max_residual)
ax1.set_xlabel("Fitted Values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residuals vs. Fitted Values")

# Normal Q-Q plot
ax2 = plt.subplot(gs[1, 1])
sm.qqplot(model.resid, line='45', fit=True, ax=ax2)
ax2.set_title("Normal Q-Q Plot")

# Scale-location plot (sqrt of standardized residuals vs. fitted values)
ax3 = plt.subplot(gs[2, 0])
ax3.scatter(model.fittedvalues, np.sqrt(np.abs(model.get_influence().resid_studentized_internal)), alpha=0.7)
ax3.set_xlabel("Fitted Values")
ax3.set_ylabel("Sqrt of Std. Residuals")
ax3.set_title("Scale-Location Plot")

# Leverage vs. standardized residuals plot
ax4 = plt.subplot(gs[2, 1])
sm.graphics.influence_plot(model, ax=ax4, size=16) # Note that points with high leverage or large residuals will be labelled
ax4.set_title("Leverage vs. Standardized Residuals", fontsize=12)
ax4.set_xlabel("Leverage", fontsize=10)
ax4.set_ylabel("Standardized Residuals", fontsize=10)

# Adjust layout and display the plots
fig.tight_layout()

####
#### Now we want to quantitatively diagnose the regression model to see if it satisfies the assumptions of linear regression
# Normality tests:
# Jarque-Bera test
alpha = 0.05
jb_name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
jb_test = sms.jarque_bera(model.resid)
print("\nNORMALITY TEST:")
print("Jarque-Bera test results:")
for name, value in lzip(jb_name, jb_test):
    print(f"{name}: {value}")
if jb_test[1] < alpha:
    print("The residuals are NOT normally distributed (reject null hypothesis)")
else:
    print("The residuals are normally distributed (fail to reject null hypothesis)")

# Omni test
omni_name = ["Chi^2", "Two-tail probability"]
omni_test = sms.omni_normtest(model.resid)
print("\nOmni test results:")
for name, value in lzip(omni_name, omni_test):
    print(f"{name}: {value}")
if omni_test[1] < alpha:
    print("The residuals are NOT normally distributed (reject null hypothesis)")
else:
    print("The residuals are normally distributed (fail to reject null hypothesis)")

# Heteroscedasticity tests:
# Breusch-Pagan test
bp_name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
bp_test = sms_diag.het_breuschpagan(model.resid, model.model.exog)
print("\nHOMOSCEDASTICITY TEST:")
print("Breusch-Pagan test results:")
for name, value in lzip(bp_name, bp_test):
    print(f"{name}: {value}")
if bp_test[1] < alpha:
    print("The residuals are NOT homoscedastic (reject null hypothesis)")
else:
    print("The residuals are homoscedastic (fail to reject null hypothesis)")

# Goldfeld-Quandt test
gq_name = ["F statistic", "p-value"]
gq_test = sms_diag.het_goldfeldquandt(model.resid, model.model.exog)
print("\nGoldfeld-Quandt test results:")
for name, value in lzip(gq_name, gq_test):
    print(f"{name}: {value}")
if gq_test[1] < alpha:
    print("The residuals are NOT homoscedastic (reject null hypothesis)")
else:
    print("The residuals are homoscedastic (fail to reject null hypothesis)")

# Linearity tests:
# Harvey-Collier multiplier test
hc_name = ["t value", "p value"]
hc_test = sms_linear_hc(model)
print("\nLINEARITY TEST:")
print("Harvey-Collier multiplier test results:")
for name, value in lzip(hc_name, hc_test):
    print(f"{name}: {value}")
if hc_test[1] < alpha:
    print("The linear specification is NOT correct (reject null hypothesis)")
else:
    print("The linear specification is correct (fail to reject null hypothesis)")

# Plot all plots
plt.show()
