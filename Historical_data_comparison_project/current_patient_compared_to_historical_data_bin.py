import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm

# Define paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
input_dir = os.path.join(parent_dir, "Data", "Historical_01", "Input")
output_dir = os.path.join(parent_dir, "Data", "Historical_01", "Output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Import data
df = pd.read_csv(os.path.join(input_dir, "Abdomen CT parameters vs Patient age, sex - out.csv"))
attr = "liver_volume" #### Change this to select different attributes

# Create the current patient's data
current_patient = {
    "patient_sex": "male",
    "patient_age": 35,
    attr: 2000,
}

# Filter the data by patient_sex, patient_age, and scan_type
min_age = 18
max_age = 100
sex = current_patient['patient_sex']
filtered_data = df.query(f'patient_age >= {min_age} and patient_age <= {max_age} and patient_sex == "{sex}" and {attr} != 0')
filtered_data = filtered_data[['patient_age', 'patient_sex', attr]]

# Calculate percentiles for each age group
age_bins = range(min_age, max_age + 1, 5)
age_bin_labels = [f"{age}-{age + 4}" for age in age_bins[:-1]]
filtered_data["age_bin"] = pd.cut(filtered_data["patient_age"], bins=age_bins, labels=age_bin_labels, right=False)
percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Set the colormap and calculate colors for percentile range
cmap = cm.get_cmap('RdYlGn')
norm = plt.Normalize(vmin=0, vmax=50)
colors = []
for p in range(0, 50, 10):
    colors.append(cmap(norm(p)))
for i in range(1, 6):
    x = colors[5-i]
    colors = np.vstack([colors, x])
#colors = ["silver", "silver", "darkgray", "limegreen", "forestgreen", "darkgreen", "darkgreen", "forestgreen", "limegreen", "darkgray", "silver"]

age_bin_percentiles = []
for age_bin in age_bin_labels:
    age_bin_data = filtered_data[filtered_data["age_bin"] == age_bin][attr]
    age_bin_percentiles.append([age_bin] + [np.percentile(age_bin_data, p) for p in percentiles])
age_bin_percentiles_df = pd.DataFrame(age_bin_percentiles, columns=["age_bin"] + percentiles)

# Convert age_bin to the middle value for plotting
age_bin_percentiles_df["mid_age"] = [int(x.split('-')[0]) + 2.5 for x in age_bin_percentiles_df["age_bin"]]

# Plot the percentile ranges with corresponding colors
plt.figure(figsize=(7, 6))
for i, p in enumerate(percentiles[:-1]):
    plt.fill_between(
        age_bin_percentiles_df["mid_age"],
        age_bin_percentiles_df[p],
        age_bin_percentiles_df[percentiles[i + 1]],
        color=colors[i],
        linewidth=0,
        alpha=0.35,
        label=f"{p}-{percentiles[i + 1]}th Percentile" if i > 0 else f"{p}-{percentiles[i + 1]}th Percentile",
    )

# Plot the scatter plot dots
sns.scatterplot(data=filtered_data, x="patient_age", y=attr, alpha=0.9, color='black', marker='.', size=1, edgecolors='none', linewidth=0, legend=False)
plt.scatter(current_patient["patient_age"], current_patient[attr], color='red')

# Plot median trendline to follow the 50th percentile mid_age points
plt.plot(age_bin_percentiles_df["mid_age"],
         age_bin_percentiles_df[50],
         color="forestgreen",
         linestyle="--",
         linewidth=2.5,
         label="Median Trendline")

# Calculate the patient's age bin
current_patient["age_bin"] = pd.cut(
    [current_patient["patient_age"]],
    bins=age_bins,
    labels=age_bin_labels,
    right=False
)[0]

# Find the age_bin data for the current patient
current_patient_age_bin_data = age_bin_percentiles_df[age_bin_percentiles_df["age_bin"] == current_patient["age_bin"]]

# Calculate outcome measurement using binned percentiles
outcome = None
for idx, p in enumerate(percentiles[:-1]):
    if current_patient[attr] <= current_patient_age_bin_data.iloc[0, idx + 1]:
        outcome = f"{percentiles[idx]}-{percentiles[idx + 1]}th Percentile"
        break
if outcome is None:
    outcome = "100th Percentile"
    
# Create the patient information box
patient_info = (
f"Current Patient:\n"
f"Sex: {current_patient['patient_sex']}\n"
f"Age: {current_patient['patient_age']}\n"
f"{attr}: {current_patient[attr]}\n"
f"Outcome: {outcome}"
)

# Determine the box color for patient info based on the outcome percentile
box_color = "white"
for idx, p in enumerate(percentiles[:-1]):
    if current_patient[attr] <= current_patient_age_bin_data.iloc[0, idx + 1]:
        box_color = colors[idx]
        break
    elif idx == len(percentiles) - 2:
        box_color = colors[-1]

# Display the patient information box below the legend
x1 = 0.4
y1 = 0.98
plt.text(
    x1,
    y1,
    patient_info,
    fontsize=9,
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor=box_color, edgecolor="black", boxstyle="round,pad=0.2", alpha=0.3),
)

# Add an arrow from the patient information box to the current patient dot
x2 = x1+0.05
y2 = y1-0.16
arrow_start = (x2, y2)
arrow_end = (current_patient["patient_age"], current_patient[attr])
plt.annotate("",
             xy=arrow_end,
             xycoords='data',
             xytext=arrow_start,
             textcoords='axes fraction',
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
             )

# Add a legend
plt.legend(loc='upper right', borderaxespad=0.2, fontsize=9)
#plt.ylim(bottom=0)
plt.xlabel("Patient Age")
plt.ylabel("Feature Size")
plt.title(f"{attr} scatterplot")
plt.tight_layout()
plt.show()
#print("fin")
