import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# Load the dataset
try:
    df = pd.read_csv('delhivery_data.csv')
    print("Dataset loaded successfully.")
    print("Shape of the dataset:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Info:")
    df.info()
    print("\nMissing values before handling:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe(include='all'))

except FileNotFoundError:
    print("Error: 'delhivery_data.csv' not found in the current directory.")
    exit()

# --- Basic Data Cleaning and Exploration ---

# 1. Handle Missing Values
# Strategy: Fill numerical NaNs with median/mean, categorical with mode or 'Unknown'.
# Let's examine the columns with missing values again.
# source_name, destination_name, od_start_time, od_end_time have few missing values.
# segment_actual_time, segment_osrm_time, segment_osrm_distance, segment_factor have many.
# is_cutoff, cutoff_factor, cutoff_timestamp have almost all missing - likely drop these.
# factor also has many missing values.

print("\nHandling Missing Values...")

# Drop columns with too many missing values
cols_to_drop = ['is_cutoff', 'cutoff_factor', 'cutoff_timestamp', 'factor', 'segment_factor']
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")

# Fill missing categorical names with 'Unknown'
df['source_name'].fillna('Unknown', inplace=True)
df['destination_name'].fillna('Unknown', inplace=True)

# Convert timestamp columns to datetime objects, coercing errors
time_cols = ['trip_creation_time', 'od_start_time', 'od_end_time']
for col in time_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Check for NaT values introduced by coercion
print("\nNaT values after datetime conversion:")
print(df[time_cols].isnull().sum())

# Fill missing timestamps - this is tricky. Filling with mean/median doesn't make sense.
# Let's drop rows where od_start_time or od_end_time is missing, as they are crucial for time calculations.
df.dropna(subset=['od_start_time', 'od_end_time'], inplace=True)
print(f"Dropped rows with missing od_start_time or od_end_time. New shape: {df.shape}")

# Fill missing numerical segment times/distances. Median might be better due to potential outliers.
num_segment_cols = ['segment_actual_time', 'segment_osrm_time', 'segment_osrm_distance']
for col in num_segment_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"Filled missing values in '{col}' with median: {median_val}")

print("\nMissing values after handling:")
print(df.isnull().sum())


# 2. Analyze Structure (already done with info(), head(), describe())

# 3. Feature Extraction

print("\nExtracting Features...")

# Destination Name: City-Place-Code (State)
# Assuming format 'City_Suffix (State)' e.g., 'Bengaluru_DC (KA)'
df[['destination_city', 'destination_state_code']] = df['destination_name'].str.extract(r'^([^_]+)_[^_]+\s+\(([^)]+)\)') # Corrected regex
df['destination_state_code'].fillna('Unknown', inplace=True) # Handle cases that didn't match
df['destination_city'].fillna(df['destination_name'], inplace=True) # Use full name if pattern fails

# Source Name: City-Place-Code (State)
df[['source_city', 'source_state_code']] = df['source_name'].str.extract(r'^([^_]+)_[^_]+\s+\(([^)]+)\)') # Corrected regex
df['source_state_code'].fillna('Unknown', inplace=True) # Handle cases that didn't match
df['source_city'].fillna(df['source_name'], inplace=True) # Use full name if pattern fails

# Trip Creation Time
df['trip_creation_year'] = df['trip_creation_time'].dt.year
df['trip_creation_month'] = df['trip_creation_time'].dt.month
df['trip_creation_day'] = df['trip_creation_time'].dt.day
df['trip_creation_hour'] = df['trip_creation_time'].dt.hour
df['trip_creation_weekday'] = df['trip_creation_time'].dt.weekday # Monday=0, Sunday=6

print("Features extracted from names and timestamps.")
print(df[['source_city', 'source_state_code', 'destination_city', 'destination_state_code', 'trip_creation_year', 'trip_creation_month', 'trip_creation_day']].head())


# --- Row Merging/Aggregation (Hint Based) ---

print("\nMerging Rows...")

# Define aggregations
# Numeric: sum for times/distances that accumulate, maybe mean for others?
# Categorical: first/last
# Timestamps: first for start, last for end

# Aggregation Level 1: trip_uuid, source_center, destination_cente
agg_funcs_level1 = {
    'data': 'first',
    'route_schedule_uuid': 'first', # Assuming constant within this group
    'route_type': 'first',
    'trip_creation_time': 'first', # Keep first creation time for the segment
    'source_name': 'first',
    'destination_name': 'first',
    'od_start_time': 'first', # Start of the first segment
    'od_end_time': 'last',   # End of the last segment within this group
    'start_scan_to_end_scan': 'sum', # Summing time for segments
    'actual_distance_to_destination': 'first', # Assuming this is overall distance, keep first
    'actual_time': 'sum',
    'osrm_time': 'sum',
    'osrm_distance': 'sum',
    'segment_actual_time': 'sum',
    'segment_osrm_time': 'sum',
    'segment_osrm_distance': 'sum',
    # Keep extracted features
    'destination_city': 'first',
    'destination_state_code': 'first',
    'source_city': 'first',
    'source_state_code': 'first',
    'trip_creation_year': 'first',
    'trip_creation_month': 'first',
    'trip_creation_day': 'first',
    'trip_creation_hour': 'first',
    'trip_creation_weekday': 'first'
}

# Grouping requires handling potential non-unique indices if any were created
df_grouped_level1 = df.groupby(['trip_uuid', 'source_center', 'destination_center'], as_index=False).agg(agg_funcs_level1) # Corrected column name

print(f"Shape after grouping by trip_uuid, source, destination: {df_grouped_level1.shape}")
print(df_grouped_level1.head())


# Aggregation Level 2: trip_uuid only
# We need to decide how to aggregate the segments for a whole trip.
# Summing times/distances makes sense. For start/end times, take the overall min/max.
# For source/destination, take the first source and last destination?

# To get the overall start/end times and first/last locations correctly, sort by time first
df_sorted = df.sort_values(by=['trip_uuid', 'od_start_time'])

agg_funcs_level2 = {
    'data': 'first',
    'route_schedule_uuid': 'first', # Assuming constant for the trip
    'route_type': 'first', # Assuming constant for the trip
    'trip_creation_time': 'first', # First creation time for the trip
    'source_center': 'first', # First source center of the trip
    'source_name': 'first', # First source name
    'destination_center': 'last', # Last destination center - Corrected column name
    'destination_name': 'last', # Last destination name
    'od_start_time': 'min', # Earliest start time
    'od_end_time': 'max',   # Latest end time
    'start_scan_to_end_scan': 'sum', # Total scan-to-scan time
    'actual_distance_to_destination': 'first', # Keep the first recorded distance? Or last? Or mean? Let's take first for now.
    'actual_time': 'sum', # Total actual time
    'osrm_time': 'sum', # Total OSRM time
    'osrm_distance': 'sum', # Total OSRM distance
    'segment_actual_time': 'sum', # Sum of segment actual times
    'segment_osrm_time': 'sum', # Sum of segment OSRM times
    'segment_osrm_distance': 'sum', # Sum of segment OSRM distances
    # Keep extracted features (first/last as appropriate)
    'destination_city': 'last',
    'destination_state_code': 'last',
    'source_city': 'first',
    'source_state_code': 'first',
    'trip_creation_year': 'first',
    'trip_creation_month': 'first',
    'trip_creation_day': 'first',
    'trip_creation_hour': 'first',
    'trip_creation_weekday': 'first'
}

df_agg_trip = df_sorted.groupby('trip_uuid', as_index=False).agg(agg_funcs_level2)

print(f"\nShape after aggregating by trip_uuid: {df_agg_trip.shape}")
print(df_agg_trip.head())
print("\nAggregated Data Info:")
df_agg_trip.info()
print("\nAggregated Data Missing Values:")
print(df_agg_trip.isnull().sum()) # Check if aggregation introduced NaNs


# --- In-depth Analysis and Feature Engineering (on df_agg_trip) ---

print("\nPerforming In-depth Analysis...")

# a. Calculate time taken between od_start_time and od_end_time
# Ensure columns are datetime
df_agg_trip['od_start_time'] = pd.to_datetime(df_agg_trip['od_start_time'])
df_agg_trip['od_end_time'] = pd.to_datetime(df_agg_trip['od_end_time'])

# Calculate difference in hours
df_agg_trip['od_time_diff_hours'] = (df_agg_trip['od_end_time'] - df_agg_trip['od_start_time']).dt.total_seconds() / 3600.0

# Drop original columns if required (optional for now)
# df_agg_trip.drop(columns=['od_start_time', 'od_end_time'], inplace=True)

print("Calculated 'od_time_diff_hours'.")
print(df_agg_trip[['od_start_time', 'od_end_time', 'od_time_diff_hours']].head())

# b. Compare od_time_diff_hours and start_scan_to_end_scan
print("\nComparing 'od_time_diff_hours' and 'start_scan_to_end_scan'...")
# Visual Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_agg_trip, x='od_time_diff_hours', y='start_scan_to_end_scan', alpha=0.5)
plt.title('od_time_diff_hours vs. start_scan_to_end_scan')
plt.xlabel('OD Start to End Time Difference (hours)')
plt.ylabel('Sum of Scan-to-Scan Time (assumed hours)') # Clarified assumed units
plt.grid(True)
plt.savefig('od_diff_vs_scan_time.png')
plt.close()
print("Saved scatter plot: od_diff_vs_scan_time.png")

# Basic statistics of the difference
df_agg_trip['scan_od_diff'] = df_agg_trip['od_time_diff_hours'] - df_agg_trip['start_scan_to_end_scan']
print("Statistics for (od_time_diff_hours - start_scan_to_end_scan):")
print(df_agg_trip['scan_od_diff'].describe())
# Hypothesis Testing (e.g., Paired t-test if we assume they measure the same underlying duration)
# Note: Requires assumptions (normality of differences). Visual inspection suggests non-normality.
# A non-parametric test like Wilcoxon signed-rank test might be more appropriate.
try:
    stat, p_value = stats.wilcoxon(df_agg_trip['od_time_diff_hours'], df_agg_trip['start_scan_to_end_scan'])
    print(f"\nWilcoxon test between od_time_diff_hours and start_scan_to_end_scan:")
    print(f"Statistic: {stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05).")
    else:
        print("No significant difference detected (p >= 0.05).")
except ValueError as e:
    print(f"\nCould not perform Wilcoxon test: {e}") # Might happen if identical values exist


# c. Compare actual_time vs OSRM time (aggregated)
print("\nComparing aggregated 'actual_time' vs 'osrm_time'...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_agg_trip, x='osrm_time', y='actual_time', alpha=0.5)
plt.title('Aggregated Actual Time vs. OSRM Time')
plt.xlabel('Aggregated OSRM Time')
plt.ylabel('Aggregated Actual Time')
plt.plot([0, df_agg_trip[['osrm_time', 'actual_time']].max().max()], [0, df_agg_trip[['osrm_time', 'actual_time']].max().max()], ls="--", c=".3") # Line y=x
plt.grid(True)
plt.savefig('actual_vs_osrm_time.png')
plt.close()
print("Saved scatter plot: actual_vs_osrm_time.png")

try:
    stat, p_value = stats.wilcoxon(df_agg_trip['actual_time'], df_agg_trip['osrm_time'])
    print(f"\nWilcoxon test between aggregated actual_time and osrm_time:")
    print(f"Statistic: {stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05).")
    else:
        print("No significant difference detected (p >= 0.05).")
except ValueError as e:
    print(f"\nCould not perform Wilcoxon test: {e}")


# d. Compare actual_time vs segment_actual_time (aggregated)
print("\nComparing aggregated 'actual_time' vs 'segment_actual_time'...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_agg_trip, x='segment_actual_time', y='actual_time', alpha=0.5)
plt.title('Aggregated Actual Time vs. Aggregated Segment Actual Time')
plt.xlabel('Aggregated Segment Actual Time')
plt.ylabel('Aggregated Actual Time')
plt.plot([0, df_agg_trip[['segment_actual_time', 'actual_time']].max().max()], [0, df_agg_trip[['segment_actual_time', 'actual_time']].max().max()], ls="--", c=".3")
plt.grid(True)
plt.savefig('actual_vs_segment_actual_time.png')
plt.close()
print("Saved scatter plot: actual_vs_segment_actual_time.png")

try:
    stat, p_value = stats.wilcoxon(df_agg_trip['actual_time'], df_agg_trip['segment_actual_time'])
    print(f"\nWilcoxon test between aggregated actual_time and segment_actual_time:")
    print(f"Statistic: {stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05).")
    else:
        print("No significant difference detected (p >= 0.05).")
except ValueError as e:
    print(f"\nCould not perform Wilcoxon test: {e}")


# e. Compare osrm_distance vs segment_osrm_distance (aggregated)
print("\nComparing aggregated 'osrm_distance' vs 'segment_osrm_distance'...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_agg_trip, x='segment_osrm_distance', y='osrm_distance', alpha=0.5)
plt.title('Aggregated OSRM Distance vs. Aggregated Segment OSRM Distance')
plt.xlabel('Aggregated Segment OSRM Distance')
plt.ylabel('Aggregated OSRM Distance')
plt.plot([0, df_agg_trip[['segment_osrm_distance', 'osrm_distance']].max().max()], [0, df_agg_trip[['segment_osrm_distance', 'osrm_distance']].max().max()], ls="--", c=".3")
plt.grid(True)
plt.savefig('osrm_dist_vs_segment_osrm_dist.png')
plt.close()
print("Saved scatter plot: osrm_dist_vs_segment_osrm_dist.png")

try:
    stat, p_value = stats.wilcoxon(df_agg_trip['osrm_distance'], df_agg_trip['segment_osrm_distance'])
    print(f"\nWilcoxon test between aggregated osrm_distance and segment_osrm_distance:")
    print(f"Statistic: {stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05).")
    else:
        print("No significant difference detected (p >= 0.05).")
except ValueError as e:
    print(f"\nCould not perform Wilcoxon test: {e}")


# f. Compare osrm_time vs segment_osrm_time (aggregated)
print("\nComparing aggregated 'osrm_time' vs 'segment_osrm_time'...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_agg_trip, x='segment_osrm_time', y='osrm_time', alpha=0.5)
plt.title('Aggregated OSRM Time vs. Aggregated Segment OSRM Time')
plt.xlabel('Aggregated Segment OSRM Time')
plt.ylabel('Aggregated OSRM Time')
plt.plot([0, df_agg_trip[['segment_osrm_time', 'osrm_time']].max().max()], [0, df_agg_trip[['segment_osrm_time', 'osrm_time']].max().max()], ls="--", c=".3")
plt.grid(True)
plt.savefig('osrm_time_vs_segment_osrm_time.png')
plt.close()
print("Saved scatter plot: osrm_time_vs_segment_osrm_time.png")

try:
    stat, p_value = stats.wilcoxon(df_agg_trip['osrm_time'], df_agg_trip['segment_osrm_time'])
    print(f"\nWilcoxon test between aggregated osrm_time and segment_osrm_time:")
    print(f"Statistic: {stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Significant difference detected (p < 0.05).")
    else:
        print("No significant difference detected (p >= 0.05).")
except ValueError as e:
    print(f"\nCould not perform Wilcoxon test: {e}")


# --- Outlier Detection and Treatment ---

print("\nHandling Outliers...")

numerical_cols = df_agg_trip.select_dtypes(include=np.number).columns.tolist()
# Remove identifier/categorical-like numeric columns if any (e.g., year, month, day)
cols_to_exclude_outliers = ['trip_creation_year', 'trip_creation_month', 'trip_creation_day', 'trip_creation_hour', 'trip_creation_weekday']
numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude_outliers]

print(f"Numerical columns for outlier check: {numerical_cols}")

# Visual Analysis (Boxplots)
plt.figure(figsize=(15, len(numerical_cols) * 2))
for i, col in enumerate(numerical_cols):
    plt.subplot(len(numerical_cols), 1, i + 1)
    sns.boxplot(x=df_agg_trip[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('boxplots_before_outlier_treatment.png')
plt.close()
print("Saved boxplots: boxplots_before_outlier_treatment.png")

# Handle outliers using IQR method
df_cleaned = df_agg_trip.copy()
outliers_info = {}

for col in numerical_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    original_count = df_cleaned.shape[0]
    # Cap the outliers instead of removing
    df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
    df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])

    outliers_count = original_count - df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)].shape[0] # Re-check count based on original bounds
    outliers_info[col] = {'lower': lower_bound, 'upper': upper_bound, 'count_capped': outliers_count}
    print(f"Capped outliers in '{col}' using IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]. Approx Capped: {outliers_count}")


print("\nOutlier capping summary:")
# print(outliers_info) # Can be verbose

# Visual Analysis After Capping
plt.figure(figsize=(15, len(numerical_cols) * 2))
for i, col in enumerate(numerical_cols):
    plt.subplot(len(numerical_cols), 1, i + 1)
    sns.boxplot(x=df_cleaned[col])
    plt.title(f'Boxplot of {col} (After Capping)')
plt.tight_layout()
plt.savefig('boxplots_after_outlier_treatment.png')
plt.close()
print("Saved boxplots after capping: boxplots_after_outlier_treatment.png")


# --- Categorical Variable Handling ---

print("\nHandling Categorical Variables...")

categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
# Also include state codes which were extracted
categorical_cols.extend(['source_state_code', 'destination_state_code'])
# Remove high cardinality columns like names/IDs if they are still object type
cols_to_exclude_encoding = ['trip_uuid', 'route_schedule_uuid', 'source_name', 'destination_name', 'source_city', 'destination_city']
categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude_encoding and col in df_cleaned.columns]

print(f"Categorical columns for encoding: {categorical_cols}")

# One-Hot Encoding
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True, dummy_na=False) # drop_first to avoid multicollinearity

print(f"Shape after One-Hot Encoding: {df_encoded.shape}")
print("Columns after encoding (sample):", df_encoded.columns[:20].tolist(), "...") # Show some new columns


# --- Normalization / Standardization ---

print("\nNormalizing/Standardizing Numerical Features...")

# Select numerical columns again from the encoded dataframe
numerical_cols_encoded = df_encoded.select_dtypes(include=np.number).columns.tolist()
# Exclude identifiers and previously excluded cols
ids_and_time_parts = ['trip_creation_year', 'trip_creation_month', 'trip_creation_day', 'trip_creation_hour', 'trip_creation_weekday'] # Keep these as they are or treat differently if needed
numerical_cols_to_scale = [col for col in numerical_cols_encoded if col not in ids_and_time_parts and col in numerical_cols] # Use original list to ensure we scale the right ones

print(f"Numerical columns to scale: {numerical_cols_to_scale}")

# Using StandardScaler
scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numerical_cols_to_scale] = scaler.fit_transform(df_scaled[numerical_cols_to_scale])

print("Applied StandardScaler to numerical features.")
print("\nScaled Data Sample (first 5 rows, selected columns):")
print(df_scaled[numerical_cols_to_scale].head())


# --- Final Data ---
print("\nFinal processed data shape:", df_scaled.shape)
# Save the processed data
df_scaled.to_csv('delhivery_data_processed.csv', index=False)
print("Saved processed data to 'delhivery_data_processed.csv'")

# --- Basic Business Insights (Example) ---
print("\n--- Basic Business Insights ---")

# 1. Most frequent routes (Source State -> Destination State)
df_agg_trip['route'] = df_agg_trip['source_state_code'] + ' -> ' + df_agg_trip['destination_state_code']
top_routes = df_agg_trip['route'].value_counts().head(10)
print("\nTop 10 Routes (Source State -> Destination State):")
print(top_routes)

# 2. Busiest Corridors (Source City -> Destination City) - High Cardinality Warning
df_agg_trip['corridor'] = df_agg_trip['source_city'] + ' -> ' + df_agg_trip['destination_city']
top_corridors = df_agg_trip['corridor'].value_counts().head(10)
print("\nTop 10 Corridors (Source City -> Destination City):")
print(top_corridors)

# 3. Average time/distance for top routes
print("\nAverage Metrics for Top 5 Routes:")
for route in top_routes.head(5).index:
    route_data = df_agg_trip[df_agg_trip['route'] == route]
    avg_actual_time = route_data['actual_time'].mean()
    avg_osrm_time = route_data['osrm_time'].mean()
    avg_osrm_dist = route_data['osrm_distance'].mean()
    avg_od_diff = route_data['od_time_diff_hours'].mean()
    print(f"\nRoute: {route}")
    print(f"  Avg Actual Time: {avg_actual_time:.2f}")
    print(f"  Avg OSRM Time: {avg_osrm_time:.2f}")
    print(f"  Avg OSRM Distance: {avg_osrm_dist:.2f}")
    print(f"  Avg Trip Duration (OD): {avg_od_diff:.2f} hours")

# 4. Distribution of Route Types
print("\nDistribution of Route Types:")
print(df_agg_trip['route_type'].value_counts(normalize=True) * 100)


print("\n--- End of Analysis ---")
