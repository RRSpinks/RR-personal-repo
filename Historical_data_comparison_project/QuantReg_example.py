import pandas as pd
from QuantReg_functions import compute_QuantReg, QuantReg_scatterplot

# Load data
data = pd.read_csv('/home/richard/Richard/RR-personal-repo/Data/Historical_01/Input/RR Image Similarity Parameters with Patient Data - Test.csv') # User to change
X = data['patient_age'] # User to change
y = data['liver_volume'] # User to change

# Example usage
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors = [  # ver 5
    [0.8, 0.0, 0.0, 0.65], # Red
    [1.0, 0.5, 0.0, 0.6],  # Orange
    [1.0, 1.0, 0.0, 0.5],  # Yellow
    [0.0, 1.0, 0.0, 0.4],  # Green
    [0.0, 0.7, 1.0, 0.5],  # Blue
    [0.0, 1.0, 0.0, 0.4],  # Green
    [1.0, 1.0, 0.0, 0.5],  # Yellow
    [1.0, 0.5, 0.0, 0.6],  # Orange
    [0.8, 0.0, 0.0, 0.65]  # Red
]
# Get quantile regression results
results = compute_QuantReg(X, y, quantiles, model_type='linear')
# Plot Quantile regression results, also conducts outlier exclusion internally
QuantReg_scatterplot(X, y, results, quantiles, colors=colors)

