import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import CubicSpline
from scipy.ndimage.filters import uniform_filter1d

np.random.seed(42)


# Custom function to generate feature size values based on age
def generate_feature_size(age, peak_age=28):
    scale_factor = peak_age / 3
    shift_factor = age - peak_age
    return np.random.gamma(age / scale_factor, scale_factor) + shift_factor

# Create the randomized dataset
data = {
    "scan_type": np.random.choice(["CT", "MRI", "Ultrasound"], 5000),
    "patient_sex": np.random.choice(["M", "F"], 5000),
    "patient_age": np.random.randint(18, 100, 5000),
}
df = pd.DataFrame(data)
df["feature_size"] = df["patient_age"].apply(generate_feature_size)

# Create the current patient's data
current_patient = {
    "scan_type": "CT",
    "patient_sex": "M",
    "patient_age": 35,
    "feature_size": 60,
}

# Filter the data by patient_sex, patient_age, and scan_type
filtered_data = df[
    (df["patient_sex"] == current_patient["patient_sex"]) &
    (df["patient_age"] >= 18) &
    (df["scan_type"] == current_patient["scan_type"])
]
# Generate scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x="patient_age", y="feature_size", alpha=0.5)
plt.scatter(current_patient["patient_age"], current_patient["feature_size"], color='red')

# Calculate KDE-based percentiles
unique_ages = np.sort(filtered_data["patient_age"].unique())
kde_results = []
for age in unique_ages:
    age_data = filtered_data[filtered_data["patient_age"] == age]["feature_size"]
    kde = gaussian_kde(age_data)
    x_vals = np.linspace(age_data.min(), age_data.max(), 100)
    y_vals = kde(x_vals)
    lower = x_vals[y_vals.cumsum() >= 0.025].min()
    upper = x_vals[y_vals.cumsum() <= 0.975].max()
    kde_results.append((age, lower, upper))

kde_percentiles = pd.DataFrame(kde_results, columns=["patient_age", "lower", "upper"])

# Apply moving average to the lower and upper percentiles
window_size = 15
kde_percentiles["smooth_lower"] = uniform_filter1d(kde_percentiles["lower"], size=window_size, mode="nearest")
kde_percentiles["smooth_upper"] = uniform_filter1d(kde_percentiles["upper"], size=window_size, mode="nearest")

# Use cubic spline interpolation for the smoothed percentiles
cs_lower = CubicSpline(kde_percentiles["patient_age"], kde_percentiles["smooth_lower"])
cs_upper = CubicSpline(kde_percentiles["patient_age"], kde_percentiles["smooth_upper"])

# Plot the smoothed percentiles as shaded areas
plt.fill_between(
    kde_percentiles["patient_age"],
    kde_percentiles["smooth_lower"],
    kde_percentiles["smooth_upper"],
    color="gray",
    alpha=0.2,
    label="95th Percentile (Smoothed)",
)
# Plot the mean trendline
mean_values = filtered_data.groupby("patient_age").mean()["feature_size"].reset_index()
window_size = 15
mean_values["smoothed_mean"] = mean_values["feature_size"].rolling(window=window_size, center=True, min_periods=1).mean()
plt.plot(mean_values["patient_age"], mean_values["feature_size"], color="blue", linestyle="--", label="Mean Trendline")

# Add a legend
plt.legend(loc='upper right', borderaxespad=0.2)
plt.ylim(bottom=0)
plt.xlabel("Patient Age")
plt.ylabel("Feature Size")
plt.title("Feature Size vs Patient Age Scatter Plot")

# Calculate outcome measurement using percentiles
patient_age_percentiles = kde_percentiles.loc[kde_percentiles["patient_age"] == current_patient["patient_age"]].iloc[0]
if current_patient["feature_size"] < cs_lower(current_patient["patient_age"]):
    outcome = "abnormally low"
elif current_patient["feature_size"] > cs_upper(current_patient["patient_age"]):
    outcome = "abnormally high"
else:
    mean_feature_size = (cs_lower(current_patient["patient_age"]) + cs_upper(current_patient["patient_age"])) / 2
    if current_patient["feature_size"] < mean_feature_size:
        outcome = "normal/low"
    elif current_patient["feature_size"] > mean_feature_size:
        outcome = "normal/high"
    else:
        outcome = "normal"

# Create the patient information box
patient_info = (
    f"Current Patient:\n"
    f"Scan Type: {current_patient['scan_type']}\n"
    f"Sex: {current_patient['patient_sex']}\n"
    f"Age: {current_patient['patient_age']}\n"
    f"Feature Size: {current_patient['feature_size']}\n"
    f"Outcome: {outcome}"
)

# Display the patient information box below the legend
plt.text(
    0.1,
    0.9,
    patient_info,
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
)

plt.show()