import numpy as np
import Similarity_test_functions as stf

# Define parameter names
params = ['age', 'liver_volume', 'liver_fat_vol', 'liver_water_vol', 'liver_stone_vol']

# Generate a random dataset of 'n' patients with parameters
n = 500
patient_data = stf.generate_data(n)

# Generate random weights for the parameters
param_weights = stf.generate_weights(params)

# Normalize and weight the dataset
weighted_data, scaler = stf.normalize_and_weight_data(patient_data, param_weights)

# Create a new patient with values for each parameter
new_patient = np.array([[60, 1500, 0.5, 0.25, 2]])

# Normalize and weight the new patient's data
normalized_new_patient = scaler.transform(new_patient)
weighted_new_patient = np.copy(normalized_new_patient)
for i, param in enumerate(params):
    weighted_new_patient[:, i] *= param_weights[param]

# Find the 10 most similar patients
k = 10
indices, similarities = stf.find_most_similar_patients(weighted_data, weighted_new_patient, k)

# Format the table data
table_data = stf.format_table_data(indices, similarities, patient_data)

# Print the results
stf.print_results(new_patient, params, table_data, k)

# Plot the comparison
stf.plot_comparison(patient_data, new_patient, table_data, params, k)
