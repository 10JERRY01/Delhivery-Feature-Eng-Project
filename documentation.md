# Delhivery Data Analysis Project Documentation

## 1. Project Objective

The primary goal of this project is to analyze logistics data from Delhivery, India's largest fully integrated logistics player. The analysis aims to clean, process, and extract meaningful features from the raw dataset (`delhivery_data.csv`) to understand delivery patterns, identify potential inefficiencies, and prepare the data for downstream tasks like building forecasting models for the data science team.

Specific objectives include:
- Cleaning and sanitizing raw data fields.
- Handling missing values and outliers.
- Engineering relevant features from existing columns (timestamps, location names).
- Aggregating segmented trip data into complete trip summaries.
- Comparing different time and distance metrics (e.g., actual vs. OSRM).
- Performing exploratory data analysis (EDA) to uncover patterns and insights.
- Preparing a final, processed dataset suitable for modeling.

## 2. Dataset Description (`delhivery_data.csv`)

The dataset contains information about individual delivery segments within larger trips. Key columns include:

- **Identifiers:** `trip_uuid`, `route_schedule_uuid`, `source_center`, `destination_center`
- **Timestamps:** `trip_creation_time`, `od_start_time`, `od_end_time`
- **Location Names:** `source_name`, `destination_name`
- **Route Type:** `route_type` (FTL, Carting)
- **Time Metrics:** `start_scan_to_end_scan`, `actual_time`, `osrm_time`, `segment_actual_time`, `segment_osrm_time`
- **Distance Metrics:** `actual_distance_to_destination`, `osrm_distance`, `segment_osrm_distance`
- **Metadata:** `data` (training/testing label - though not used in this analysis phase)
- **Unknown/Dropped:** `is_cutoff`, `cutoff_factor`, `cutoff_timestamp`, `factor`, `segment_factor` (dropped due to excessive missing values or unclear meaning)

*Note: A single trip (`trip_uuid`) can consist of multiple rows (segments) in the original dataset.*

## 3. Analysis Steps (`analysis.py`)

The core analysis is performed by the `analysis.py` script. Here's a breakdown of the steps:

1.  **Load Data:** Reads `delhivery_data.csv` into a pandas DataFrame.
2.  **Initial Exploration:** Prints basic information (shape, head, info, missing values, statistical summary).
3.  **Missing Value Handling:**
    *   Drops columns with a high percentage of missing values (`is_cutoff`, `cutoff_factor`, `cutoff_timestamp`, `factor`, `segment_factor`).
    *   Fills missing `source_name` and `destination_name` with 'Unknown'.
    *   Converts timestamp columns (`trip_creation_time`, `od_start_time`, `od_end_time`) to datetime objects.
    *   Drops rows where `od_start_time` or `od_end_time` are missing (essential for duration calculations).
    *   Fills missing numerical segment times/distances (`segment_actual_time`, `segment_osrm_time`, `segment_osrm_distance`) with their respective medians.
4.  **Feature Extraction:**
    *   Extracts `destination_city` and `destination_state_code` from `destination_name` using regex `r'^([^_]+)_[^_]+\s+\(([^)]+)\)'`. Handles non-matches.
    *   Extracts `source_city` and `source_state_code` from `source_name` using the same regex pattern. Handles non-matches.
    *   Extracts temporal features (`year`, `month`, `day`, `hour`, `weekday`) from `trip_creation_time`.
5.  **Data Aggregation:**
    *   Sorts the DataFrame by `trip_uuid` and `od_start_time` to ensure correct ordering for aggregation.
    *   Groups the data by `trip_uuid` to consolidate segments into single trips.
    *   Applies aggregation functions:
        *   `sum` for cumulative metrics (`start_scan_to_end_scan`, `actual_time`, `osrm_time`, `osrm_distance`, segment times/distances).
        *   `min` for the earliest `od_start_time`.
        *   `max` for the latest `od_end_time`.
        *   `first` for initial trip details (source info, creation time, route type, etc.).
        *   `last` for final trip details (destination info).
    *   Creates the aggregated DataFrame `df_agg_trip`.
6.  **Trip Duration Calculation:** Calculates `od_time_diff_hours` (difference between max `od_end_time` and min `od_start_time` in hours) for each aggregated trip.
7.  **Comparative Analysis & Visualization:**
    *   Compares `od_time_diff_hours` vs. `start_scan_to_end_scan` (summed).
    *   Compares `actual_time` (summed) vs. `osrm_time` (summed).
    *   Compares `actual_time` (summed) vs. `segment_actual_time` (summed).
    *   Compares `osrm_distance` (summed) vs. `segment_osrm_distance` (summed).
    *   Compares `osrm_time` (summed) vs. `segment_osrm_time` (summed).
    *   For each comparison:
        *   Generates a scatter plot saved as a PNG file (e.g., `actual_vs_osrm_time.png`).
        *   Performs a Wilcoxon signed-rank test to check for significant differences.
8.  **Outlier Treatment:**
    *   Identifies numerical columns in the aggregated data.
    *   Visualizes potential outliers using boxplots (saved as `boxplots_before_outlier_treatment.png`).
    *   Applies capping using the IQR (Interquartile Range) method: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are replaced with the respective boundary values.
    *   Visualizes data after capping using boxplots (saved as `boxplots_after_outlier_treatment.png`).
9.  **Categorical Variable Handling:**
    *   Identifies categorical columns (including extracted state codes).
    *   Applies one-hot encoding using `pd.get_dummies`, dropping the first category to avoid multicollinearity.
10. **Normalization/Standardization:**
    *   Identifies numerical columns suitable for scaling (excluding time components like year, month, day).
    *   Applies `StandardScaler` to standardize these features (mean=0, stddev=1).
11. **Save Processed Data:** Saves the final, processed DataFrame (`df_scaled`) to `delhivery_data_processed.csv`.
12. **Basic Business Insights:** Calculates and prints:
    *   Top 10 routes (Source State -> Destination State).
    *   Top 10 corridors (Source City -> Destination City).
    *   Average time/distance metrics for the top 5 routes.
    *   Distribution of route types (FTL vs. Carting).

## 4. Key Findings & Insights (from script output)

- **Data Quality:** The original dataset had missing values, particularly in segment-level fields and some location names. Several columns were almost entirely empty or had unclear meanings and were dropped. Timestamps required conversion.
- **Aggregation:** Aggregating data by `trip_uuid` significantly reduced the number of rows (from ~144k to ~14.8k), providing one record per complete trip.
- **Time/Distance Comparisons:** Statistical tests (Wilcoxon) consistently showed significant differences (p < 0.05) between:
    *   Actual time/distance and OSRM estimates.
    *   Aggregated trip metrics (`actual_time`, `osrm_time`, etc.) and the sum of their corresponding segment metrics. This suggests that simply summing segment data might not perfectly represent the overall trip, potentially due to dwell times or other factors not captured in segments.
    *   The calculated `od_time_diff_hours` and the summed `start_scan_to_end_scan`, indicating these likely measure different aspects of the trip duration.
- **Outliers:** Numerical features exhibited outliers, which were handled by capping using the IQR method.
- **Feature Importance (Implied):** The high number of features generated after one-hot encoding (~1900 columns) highlights the importance of dimensionality reduction or feature selection in subsequent modeling steps.
- **Business Insights:**
    *   The majority of routes have 'Unknown' source or destination states, indicating issues with the state extraction logic or inconsistencies in the original `source_name`/`destination_name` format for many entries. This needs further investigation if state-level analysis is critical.
    *   Carting is the more frequent `route_type` (~60%) compared to FTL (~40%) in the aggregated trips.
    *   Specific city-to-city corridors show high frequency (e.g., within Chandigarh, Bangalore, Muzaffarpur).

## 5. Output Files

- `analysis.py`: The Python script containing all analysis steps.
- `delhivery_data_processed.csv`: The final cleaned, aggregated, and scaled dataset.
- `*.png`: Various plots generated during the analysis:
    - `od_diff_vs_scan_time.png`
    - `actual_vs_osrm_time.png`
    - `actual_vs_segment_actual_time.png`
    - `osrm_dist_vs_segment_osrm_dist.png`
    - `osrm_time_vs_segment_osrm_time.png`
    - `boxplots_before_outlier_treatment.png`
    - `boxplots_after_outlier_treatment.png`

## 6. Potential Next Steps & Recommendations

- **Investigate 'Unknown' States:** Determine why state extraction failed for a large portion of the data. This might involve refining the regex, manual correction, or alternative geocoding methods if possible.
- **Refine Feature Engineering:** Explore more sophisticated features, such as time-of-day categories, day-of-week interactions, or route complexity metrics.
- **Analyze Time Differences:** Deep dive into the discrepancies between actual time, OSRM time, and segment times. Identify factors contributing to delays (e.g., specific routes, times of day, route types).
- **Modeling:** Use the `delhivery_data_processed.csv` dataset to build predictive models (e.g., predicting `actual_time` or delays). Feature selection will be crucial given the high dimensionality after encoding.
- **Business Recommendations:**
    *   Focus on optimizing high-frequency corridors identified in the analysis.
    *   Investigate routes where the difference between actual time and OSRM time is consistently large, as these may represent opportunities for operational improvements or better estimation.
    *   Improve data capture quality for location names to enable more reliable geographical analysis.
