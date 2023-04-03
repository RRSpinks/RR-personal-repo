import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import CubicSpline
from scipy.ndimage.filters import uniform_filter1d

# Define paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
input_dir = os.path.join(parent_dir, "Data", "Historical_01", "Input")
output_dir = os.path.join(parent_dir, "Data", "Historical_01", "Output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Import data
df = pd.read_csv(os.path.join(input_dir, "Abdomen CT parameters vs Patient age, sex - out.csv"))

#np.random.seed(42)
## Custom function to generate feature size values based on age
#def generate_feature_size(age, peak_age=28):
#    scale_factor = peak_age / 3
#    shift_factor = age - peak_age
#    return np.random.gamma(age / scale_factor, scale_factor) + shift_factor

## Create the randomized dataset
#data = {
#    "scan_type": np.random.choice(["CT", "MRI", "Ultrasound"], 5000),
#    "patient_sex": np.random.choice(["M", "F"], 5000),
#    "patient_age": np.random.randint(18, 100, 5000),
#}
#df = pd.DataFrame(data)
#f["feature_size"] = df["patient_age"].apply(generate_feature_size)

attr = "liver_volume"

# Create the current patient's data
current_patient = {
    "scan_type": "CT",
    "patient_sex": "M",
    "patient_age": 35,
    attr: 60,
}

# Filter the data by patient_sex, patient_age, and scan_type
filtered_data = df[
    (df["patient_sex"] == current_patient["patient_sex"]) &
    (df["patient_age"] >= 18)]
filtered_data = filtered_data["patient_age", "patient_sex", attr]

# Calculate percentiles for each age group
percentiles = [0, 20, 40, 60, 80, 100]
colors = ['indianred', 'indianred', 'gold', 'limegreen', 'gold', 'indianred']
unique_ages = np.sort(filtered_data["patient_age"].unique())
age_percentiles = []
for age in unique_ages:
    age_data = filtered_data[filtered_data["patient_age"] == age][attr]
    age_percentiles.append([age] + [np.percentile(age_data, p) for p in percentiles])
age_percentiles_df = pd.DataFrame(age_percentiles, columns=["patient_age"] + percentiles)

# Apply moving average to the percentiles
#window_size = 15
#for p in percentiles:
#    age_percentiles_df[f"smooth_{p}"] = uniform_filter1d(age_percentiles_df[p], size=window_size, mode="nearest")

# Plot the percentile ranges with corresponding colors
plt.figure(figsize=(7, 6))
for i, p in enumerate(percentiles[:-1]):
    plt.fill_between(
        age_percentiles_df["patient_age"],
        age_percentiles_df[p],
        age_percentiles_df[percentiles[i + 1]],
        color=colors[i + 1],
        linewidth=0,
        alpha=0.4,
        label=f"{p}-{percentiles[i + 1]}th Percentile" if i > 0 else f"{p}-{percentiles[i + 1]}th Percentile",
    )

# Create a dictionary of cubic spline interpolation functions for each smoothed percentile
cs_dict = {}
for p in percentiles:
    cs_dict[p] = CubicSpline(age_percentiles_df["patient_age"], age_percentiles_df[p])

# Plot the scatter plot dots
sns.scatterplot(data=filtered_data, x="patient_age", y=attr, alpha=0.4, color='grey', edgecolors='none', linewidth=0)
plt.scatter(current_patient["patient_age"], current_patient[attr], color='red')

# Calculate the median trendline
median_values = filtered_data.groupby("patient_age").median()[attr].reset_index()
window_size = 10
median_values["smoothed_median"] = median_values[attr].rolling(window=window_size, center=True, min_periods=1).median()

# Plot the median trendline
plt.plot(median_values["patient_age"],
         median_values["smoothed_median"],
         color="black",
         linestyle="--",
         linewidth=1.2,
         label="Median Trendline")

# Calculate outcome measurement using percentiles
current_patient_percentiles = [cs_dict[x](current_patient["patient_age"]) for x in percentiles]
outcome = None
for idx, value in enumerate(current_patient_percentiles):
    if current_patient[attr] <= value:
        outcome = f"{percentiles[idx]}th Percentile"
        break
if outcome is None:
    outcome = "100th Percentile"
    
# Create the patient information box
patient_info = (
f"Current Patient:\n"
f"Scan Type: {current_patient['scan_type']}\n"
f"Sex: {current_patient['patient_sex']}\n"
f"Age: {current_patient['patient_age']}\n"
f"Feature Size: {current_patient['feature_size']}\n"
f"Outcome: {outcome}"
)

# Determine the box color for patient info based on the outcome percentile
box_color = "white"
for idx, value in enumerate(current_patient_percentiles):
    if current_patient[attr] <= value:
        box_color = colors[idx + 1]
        break

# Display the patient information box below the legend
x1 = 0.3
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
y2 = y1-0.18
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
plt.legend(loc='upper left', borderaxespad=0.2, fontsize=9)
#plt.ylim(bottom=0)
plt.xlabel("Patient Age")
plt.ylabel("Feature Size")
plt.title("Feature Size vs Patient Age")

plt.tight_layout()
plt.show()