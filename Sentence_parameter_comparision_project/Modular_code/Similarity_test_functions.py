import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


def generate_data(n, seed=42):
    np.random.seed(seed)

    patient_age = np.random.randint(20, 100, size=(n, 1))
    liver_volume = np.random.randint(800, 3500, size=(n, 1))
    liver_fat_vol = np.random.rand(n, 1)
    liver_water_vol = np.random.rand(n, 1)
    liver_stone_vol = np.random.randint(0, 10, size=(n, 1))

    return np.hstack((patient_age, liver_volume, liver_fat_vol, liver_water_vol, liver_stone_vol))


def generate_weights(params, seed=42):
    np.random.seed(seed)

    weights = np.random.rand(len(params))
    weights /= np.sum(weights)

    return {param: weight for param, weight in zip(params, weights)}


def normalize_and_weight_data(data, param_weights):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    weighted_data = np.copy(normalized_data)
    for i, param in enumerate(param_weights.keys()):
        weighted_data[:, i] *= param_weights[param]

    return weighted_data, scaler


def find_most_similar_patients(weighted_data, weighted_new_patient, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nbrs.fit(weighted_data)
    distances, indices = nbrs.kneighbors(weighted_new_patient)

    return indices, np.exp(-distances)


def format_table_data(indices, similarities, patient_data):
    table_data = []
    for i, (index, similarity) in enumerate(zip(indices[0], similarities[0]), start=1):
        row = [i, index]
        row.extend(patient_data[index])
        row.append(round(similarity * 100, 2))
        table_data.append(row)

    return table_data


def print_results(new_patient, params, table_data, k):
    print('=' * 40)
    print("New patient:")
    for param, value in zip(params, new_patient[0]):
        print(f"{param.capitalize()}: {value}")

    headers = ['Rank', 'Accession #'] + [param.capitalize() for param in params] + ['Similarity (%)']
    print(f"Top {k} most similar patients:")
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))


def plot_comparison(patient_data, new_patient, params, table_data, k):
    n_rows, n_cols = 2, 2
    params_to_plot = [param for param in params if param != 'age']
    fig_idx = 0

    for param_idx, param in enumerate(params_to_plot):
        if param_idx % (n_rows * n_cols) == 0:
            fig_idx += 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))

        i, j = (param_idx // n_cols) % n_rows, param_idx % n_cols
        ax = axes[i, j]

        # Plot dataset
        ax.scatter(patient_data[:, 0], patient_data[:, params.index(param)], label='Data', marker='o', color='grey', alpha=0.4)

        # Plot new patient
        ax.scatter(new_patient[0, 0], new_patient[0, params.index(param)], label='New Patient', marker='x', color='red', s=100)

        # Plot the top 10 most similar patients with a blue gradient
        cmap = plt.get_cmap('Blues_r')
        norm = mcolors.Normalize(vmin=0, vmax=k * 1.25)
        top_age = [row[2] for row in table_data]

        for rank, row in enumerate(table_data, start=1):
            age = row[2]  # age is always in the third position in the row
            label = f"P-{rank} ({row[-1]}%)"  # the similarity is always at the end of the row

            # Now we find the value for the parameter we're currently plotting.
            # We need to add 2 to the index because the first three columns are Rank, Accession # and Age.
            value = row[params.index(param) + 2]
            ax.scatter(age, value, marker='o', color=cmap(norm(rank)), alpha=0.8, label=label)

        ax.set_xlabel('Age')
        ax.set_ylabel(param)
        ax.set_title(f'Age vs {param}')
        ax.legend(loc='upper right', borderaxespad=0.05)

    plt.tight_layout()
    plt.show()


