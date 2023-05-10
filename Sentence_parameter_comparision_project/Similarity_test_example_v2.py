import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# =================================================
# INSTRUCTIONS
# =================================================
# This script finds the top 10 most similar patients based on age and liver volume 
# from a dataset of random patients. 
# It uses a weighted Euclidean distance measure and the k-Nearest Neighbors algorithm 
# to compare a new patient's age and liver volume to the dataset. 
# The output is a summary table and comparison graphs.
# =================================================

# =================================================
# IMPORT/GENERATE DATA
# =================================================
# Define random seed for consistency
np.random.seed(42)

# Define parameter names and weights
params = ['age', 'p1', 'p2', 'p3', 'p4']
weights = [500, 10, 10, 10, 10] #### MODIFY AS YOU SEE FIT (higher weight means more influence on similarity)
weights = weights / np.sum(weights)  # Normalize the weights so they sum up to 1
param_weights = dict(zip(params, weights))

# Generate a random dataset of 'n'' patients with parameters
n = 300 # no. of patients
patient_age = np.random.randint(20, 100, size=(n, 1))
p1 = np.random.randint(800, 3500, size=(n, 1)) # meant to be liver volume
p2 = np.random.rand(n, 1) # meant to be liver fat volume
p3 = np.random.rand(n, 1) # meant to be liver water volume
p4 = np.random.randint(0, 10, size=(n, 1)) # meant to be liver stone volume
patient_data = np.hstack((patient_age, p1, p2, p3, p4))

# =================================================
# MANIPULATE DATA
# =================================================
# Normalize the dataset
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(patient_data)

# Apply the weights to the normalized dataset
weighted_data = np.copy(normalized_data)
for i, param in enumerate(params):
    weighted_data[:, i] *= param_weights[param]

# Create a new patient with values for each parameter
new_patient = np.array([[60, 1500, 0.5, 0.25, 2]])

# Normalize the new patient's data
normalized_new_patient = scaler.transform(new_patient)

# Apply the weights to the new patient's data
weighted_new_patient = np.copy(normalized_new_patient)
for i, param in enumerate(params):
    weighted_new_patient[:, i] *= param_weights[param]

# Find the 'k' most similar patients using k-Nearest Neighbors
k = n  # change this to your desired number
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
nbrs.fit(weighted_data)
distances, indices = nbrs.kneighbors(weighted_new_patient)

# Convert distances to similarity values
similarities = np.exp(-distances)

# Create a table with the k most similar patients and their similarity percentages
table_data_all = []
for i, (index, similarity) in enumerate(zip(indices[0], similarities[0]), start=1):
    row = [i, index]
    row.extend(patient_data[index])
    row.append(round(similarity * 100, 2))
    table_data_all.append(row)

# =================================================
# PRINT RESULTS
# =================================================
# new patient
print('=' * 40)
print("New patient:")
for param, value in zip(params, new_patient[0]):
    print(f"{param.capitalize()}: {value}")
# Most similar patients
# Create a table with the most similar patients and their similarity percentages
# Sort table_data by similarity and take top 'p'
p = 10  # change this to your desired number
table_data = sorted(table_data_all, key=lambda x: x[-1], reverse=True)[:p]

headers = ['Rank', 'Accession #'] + [param.capitalize() for param in params] + ['Similarity (%)']
print(f"Top {p} most similar patients out of {k} patients:")
print(tabulate(table_data, headers=headers, tablefmt='pretty'))

# =================================================
# PLOT RESULTS
# =================================================
# Extract age and values of the top 10 most similar patients
top_values = {'age': [row[headers.index('Age')] for row in table_data]}
for param in params[1:]:  # ignore 'age'
    top_values[param] = [row[headers.index(param.capitalize())] for row in table_data]

# Extract similarity scores of the top 10 most similar patients
top_values['similarity'] = [row[headers.index('Similarity (%)')] for row in table_data]

# Figure 1: Multi-subplot of Age vs Parameters
params_to_plot = params[1:]  # ignore 'age'
n_rows, n_cols = 2, 2
fig_idx = 0
for param_idx, param in enumerate(params_to_plot):
    if param_idx % (n_rows * n_cols) == 0:
        fig_idx += 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))

    i, j = (param_idx // n_cols) % n_rows, param_idx % n_cols
    ax = axes[i, j]

    # Plot dataset
    ax.scatter(patient_data[:, 0], patient_data[:, params.index(param)], label='Data', marker='o', color='grey', alpha=0.4)

    # Plot new patient
    ax.scatter(new_patient[0, 0], new_patient[0, params.index(param)], label='New Patient', marker='x', color='red', s=100)

    # Plot the top 'k' most similar patients with a colour gradient
    cmap = plt.get_cmap('summer')
    norm = mcolors.Normalize(vmin=0, vmax=p * 1)
    for rank, value in enumerate(zip(top_values['age'], top_values[param]), start=1):
        age, param_value = value
        label = f"P-{rank} ({round(table_data[rank-1][-1], 1)}%)"
        ax.scatter(age, param_value, marker='o', color=cmap(norm(rank)), alpha=0.8, label=label)

    ax.set_xlabel('Age')
    ax.set_ylabel(param.capitalize())
    ax.set_title(f'Age vs {param.capitalize()}')
    ax.legend(loc='upper right', borderaxespad=0.05)

plt.tight_layout()
plt.show()
