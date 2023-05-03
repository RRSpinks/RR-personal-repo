import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.random as rnd

# Load data
def load_data(data_type):
    rnd.seed(5)
    # Choose the dataset:
        # 0 = User
        # 1 = Real life example, suits linear regression
        # 2 = Synthetic example, suits linear regression
        # 3 = Synthetic example, normality issue
        # 4 = Synthetic example, homoscedasticity issue
        # 5 = Synthetic example, linearity issue
    if data_type == 0:
        # User 
        data = pd.read_csv('/home/richard/Richard/RR-personal-repo/Data/Historical_01/Input/RR Image Similarity Parameters with Patient Data - Test.csv') # User to change
        X = data['patient_age'] # User to change
        y = data['liver_volume'] # User to change
    elif data_type == 1:
        # Example dataset 1
        data = sm.datasets.get_rdataset("longley", "datasets").data
        X = data['GNP.deflator']
        y = data['Employed']
    elif data_type == 2:
        # Example dataset 2  (This is synthetic data that satisfies all assumptions for linear regression)
        x = np.linspace(0, 10, 100)
        y = 3 * x + 5 + rnd.normal(0, 0.5, 100)  # some added noise
        data = pd.DataFrame({'X': x, 'Y': y})
        X = data['X'].reset_index(drop=True)
        y = data['Y'].reset_index(drop=True)
    elif data_type == 3:
        # Example dataset 3  (This is synthetic data that violates the normality assumptions of linear regression)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 3 + rnd.lognormal(0, 0.5, 100)  # Linear relationship with log-normal noise
        data = pd.DataFrame({'X': x, 'Y': y})
        X = data['X'].reset_index(drop=True)
        y = data['Y'].reset_index(drop=True)
    elif data_type == 4:
        # Example dataset 4  (This is synthetic data that violates the homoscedasticity assumptions of linear regression)
        x = np.linspace(0, 10, 100)
        y = 2 * x + rnd.normal(0, x, 100)  # Linear relationship with increasing (heteroscedastic) noise, note: will also be non-normal distribution
        data = pd.DataFrame({'X': x, 'Y': y})
        X = data['X'].reset_index(drop=True)
        y = data['Y'].reset_index(drop=True)
    elif data_type == 5:
        # Example dataset 5  (This is synthetic data that violates the linearity assumptions of linear regression)
        x = np.linspace(0, 10, 100)
        y = x**2 + rnd.normal(0, 1, 100)  # Quadratic relationship, note: will also be non-normal distribution for linear, but would be normal if it was a non-linear model
        data = pd.DataFrame({'X': x, 'Y': y})
        X = data['X'].reset_index(drop=True)
        y = data['Y'].reset_index(drop=True)
    # Identify and exclude outliers
    m = 3 # outlier exclusion multiplier, default = 1.5, higher = less exclusion, lower = less exclusion
    Q1_y = y.quantile(0.25)
    Q3_y = y.quantile(0.75)
    IQR_y = Q3_y - Q1_y
    if X.dtype != 'int64': # Only exclude outliers for X if the data is NOT discrete (i.e. don't exclude age data)
        Q1_x = X.quantile(0.25)
        Q3_x = X.quantile(0.75)
        IQR_x = Q3_x - Q1_x
    else:
        IQR_x = None
    outliers_y = ((y < (Q1_y - m * IQR_y)) | (y > (Q3_y + m * IQR_y)))
    if IQR_x is not None:
        outliers_x = ((X < (Q1_x - m * IQR_x)) | (X > (Q3_x + m * IQR_x)))
    else:
        outliers_x = pd.Series([False] * len(X))
    outliers = outliers_y | outliers_x
    X_no_outliers = X[~outliers].reset_index(drop=True)
    y_no_outliers = y[~outliers].reset_index(drop=True)

    return X, y, X_no_outliers, y_no_outliers, outliers

# Quantile regression
def quantile_regression(X, y, quantiles):
    data = pd.concat([X, y], axis=1)
    data.columns = ['X', 'y']
    results = []
    for q in quantiles:
        model = smf.quantreg('y ~ X', data).fit(q=q)
        results.append(model)

    return results

# Diagnostic plots
def diagnostic_plots(X, y, X_no_outliers, y_no_outliers, outliers, results, quantiles, x_label, y_label):
    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1])
    # Scatterplot of original data with the quantile regression models
    x_values = np.linspace(min(X), max(X), 100)
    ax0 = plt.subplot(gs[0, :])
    ax0.scatter(X_no_outliers, y_no_outliers, alpha=0.5, label="Data")
    ax0.scatter(X[outliers], y[outliers], alpha=0.5, label="Outliers", c='r', marker='x')  # Plot excluded outliers
    for q, model in zip(quantiles, results):
        y_values = model.params[0] + model.params[1] * x_values
        ax0.plot(x_values, y_values, label=f"{int(q*100)}th Quantile")
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)
    ax0.set_title("Scatterplot of Original Data with Quantile Regression Models")
    ax0.legend()
    # Residuals vs. fitted values plot
    ax1 = plt.subplot(gs[1, 0])
    ax1.axhline(y=0, color='red', linestyle='-')
    max_residual = 0
    for q, model in zip(quantiles, results):
        ax1.scatter(model.fittedvalues, model.resid, alpha=0.5, label=f"{int(q*100)}th Quantile")
        max_residual = max(max_residual, np.max(np.abs(model.resid)))
    ax1.set_ylim(-1.1*max_residual, 1.1*max_residual)  # Set the y-axis limits
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs. Fitted Values")
    ax1.legend()
    fig.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    data_type = 0  # Choose the type of data: 0 = User, 1 = Example 1, etc.
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Define quantiles for quantile regression
    X, y, X_no_outliers, y_no_outliers, outliers = load_data(data_type)
    results = quantile_regression(X_no_outliers, y_no_outliers, quantiles)
    for q, model in zip(quantiles, results):
        print(f"{int(q*100)}th Quantile Regression Model Summary")
        print("=" * 40)
        print(model.summary())
    diagnostic_plots(X, y, X_no_outliers, y_no_outliers, outliers, results, quantiles, X.name, y.name)


