import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# =================================================
# This script finds the top 10 most similar patients based on age and liver volume 
# from a dataset of 500 random patients. 
# It uses a weighted Euclidean distance measure and the k-Nearest Neighbors algorithm 
# to compare a new patient's age and liver volume to the dataset. 
# The output is a summary table and comparison graph.
# =================================================


# Generate a random dataset of 500 patients with age and liver volume
np.random.seed(42)
patient_age = np.random.randint(20, 100, size=(500, 1))
liver_volume = np.random.randint(800, 3500, size=(500, 1))
patient_data = np.hstack((patient_age, liver_volume))

# Normalize the dataset
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(patient_data)

# Set the weights for age and liver volume
w_age, w_liver_volume = 0.3, 0.7

# Apply the weights to the normalized dataset
weighted_data = np.copy(normalized_data)
weighted_data[:, 0] *= w_age
weighted_data[:, 1] *= w_liver_volume

# Create a new patient with age and liver volume
new_patient = np.array([[60, 1500]])

# Normalize the new patient's data
normalized_new_patient = scaler.transform(new_patient)

# Apply the weights to the new patient's data
weighted_new_patient = np.copy(normalized_new_patient)
weighted_new_patient[:, 0] *= w_age
weighted_new_patient[:, 1] *= w_liver_volume

# Find the 10 most similar patients using k-Nearest Neighbors
k = 10
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
nbrs.fit(weighted_data)
distances, indices = nbrs.kneighbors(weighted_new_patient)

# Convert distances to similarity values
similarities = np.exp(-distances)

# Create a table with the most similar patients and their similarity percentages
table_data = []
for i, (index, similarity) in enumerate(zip(indices[0], similarities[0]), start=1):
    table_data.append([i, index, patient_data[index, 0], patient_data[index, 1], round(similarity * 100, 2)])

# Print the new patient's details
print('=' * 40)
print(f"New patient: Age {new_patient[0, 0]}, Liver Volume {new_patient[0, 1]}")

# Print the most similar patients as a table
headers = ['Rank', 'Accession #', 'Age', 'Liver Volume', 'Similarity (%)']
print(f"Top {k} most similar patients:")
print(tabulate(table_data, headers=headers, tablefmt='pretty'))

# Extract age and liver volume of the top 10 most similar patients
top_age = [row[2] for row in table_data]
top_liver_volume = [row[3] for row in table_data]


# Create a scatter plot for the dataset with new patient and most similar patients
plt.scatter(patient_data[:, 0], patient_data[:, 1], label='Data', marker='o', color='grey', alpha=0.4)
plt.scatter(new_patient[0, 0], new_patient[0, 1], label='New Patient', marker='x', color='red', s=100)  # Plot the new patient in red

# Plot the top 10 most similar patients with a blue gradient
cmap = plt.get_cmap('Blues_r')
norm = mcolors.Normalize(vmin=0, vmax=k * 1.25)
for i, (age, lv) in enumerate(zip(top_age, top_liver_volume), start=1):
    label = f"P-{i} ({round(table_data[i-1][-1], 1)}%)" 
    plt.scatter(age, lv, marker='o', color=cmap(norm(i)), alpha=0.8, label=label)

plt.xlabel('Age')
plt.ylabel('Liver Volume')
plt.title('Age vs Liver Volume')
plt.legend(loc='upper right', borderaxespad=0.05)
plt.show()