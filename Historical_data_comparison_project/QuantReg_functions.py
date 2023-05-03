import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def compute_QuantReg(X, y, quantiles, model_type='linear'):
    data = pd.concat([X, y], axis=1)
    data.columns = ['X', 'y']
    results = []
    for q in quantiles:
        if model_type == 'linear':
            model = smf.quantreg('y ~ X', data).fit(q=q)
        elif model_type == 'polynomial':
            degree = 2
            model = smf.quantreg(f'y ~ X + I(X**{int(degree)})', data).fit(q=q)
        else:
            raise ValueError("Invalid model type. Choose either 'linear' or 'polynomial'.")
        results.append(model)

    return results

def exclude_outliers(X, y):
    m = 2.8 # adjust this to change outlier tolerance
    Q1_y = y.quantile(0.25)
    Q3_y = y.quantile(0.75)
    IQR_y = Q3_y - Q1_y
    if X.dtype != 'int64':
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
    X_no_outliers = X.loc[~outliers]
    y_no_outliers = y.loc[~outliers]
    
    return X_no_outliers, y_no_outliers, outliers

def QuantReg_scatterplot(X, y, results, quantiles, colors, print_summary=False):
    X_no_outliers, y_no_outliers, outliers = exclude_outliers(X, y)
    x_label = X.name
    y_label = y.name
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = np.linspace(min(X), max(X), 100)
    ax.scatter(X_no_outliers, y_no_outliers, alpha=0.5, s=22, label="Data")
    ax.scatter(X[outliers], y[outliers], alpha=0.5, s=22, label="Outliers", c='r', marker='x')
    # Fit and plot OLS linear regression as well
    cleaned_data = pd.concat([X_no_outliers, y_no_outliers], axis=1).dropna()
    X_cleaned = cleaned_data[X_no_outliers.name]
    y_cleaned = cleaned_data[y_no_outliers.name]
    ols_model = sm.OLS(y_cleaned, sm.add_constant(X_cleaned)).fit()
    ax.plot(x_values, ols_model.params[0] + ols_model.params[1] * x_values, label='OLS Linear Regression', c='k', linestyle='dashed', linewidth=1.5)
    for i, (q, model) in enumerate(zip(quantiles, results)):
        color = colors[i % len(colors)]
        if len(model.params) == 2:
            y_values = model.params[0] + model.params[1] * x_values
            equation = f"{model.params[1]:.2f}x + {model.params[0]:.2f}"
        elif len(model.params) == 3:
            y_values = model.params[0] + model.params[1] * x_values + model.params[2] * (x_values ** 2)
            equation = f"{model.params[2]:.2f}x^2 + {model.params[1]:.2f}x + {model.params[0]:.2f}"
        ax.plot(x_values, y_values, color=color, label=f"{int(q*100)}th Pctl, {equation}")
        if print_summary:
            print(f"{int(q * 100)}th Quantile Regression Model Summary")
            print("=" * 40)
            print(model.summary())
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.legend(fontsize=8)
    plt.text(0.05, 0.9, f"OLS R^2: {ols_model.rsquared:.4f}", fontsize=10, transform=ax.transAxes)
    plt.tight_layout()
    plt.show()


