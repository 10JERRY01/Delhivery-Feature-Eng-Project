# Delhivery Logistics Data Analysis

## Overview

This project analyzes logistics data from Delhivery (`delhivery_data.csv`). The goal is to clean the data, perform exploratory data analysis (EDA), engineer relevant features, aggregate trip segments, handle outliers, and prepare a processed dataset suitable for further analysis or machine learning tasks.

The analysis focuses on understanding trip durations, comparing actual vs. estimated (OSRM) time/distance, and identifying patterns in the logistics network.

## Files

- `delhivery_data.csv`: The original raw dataset (provided separately).
- `analysis.py`: The main Python script performing all data processing and analysis steps.
- `delhivery_data_processed.csv`: The output dataset after cleaning, aggregation, feature engineering, outlier capping, encoding, and scaling.
- `documentation.md`: Detailed documentation explaining the project objectives, data, methodology, findings, and potential next steps.
- `*.png`: Image files containing plots generated during the analysis (boxplots for outliers, scatter plots for comparisons).

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## How to Run

1.  **Place Data:** Ensure the `delhivery_data.csv` file is in the same directory as the `analysis.py` script.
2.  **Execute Script:** Run the Python script from your terminal:
    ```bash
    python analysis.py
    ```
3.  **Outputs:** The script will:
    *   Print progress, summaries, and basic insights to the console.
    *   Generate the `delhivery_data_processed.csv` file.
    *   Generate several `.png` plot files in the same directory.

## Analysis Steps Summary

The `analysis.py` script performs the following major steps:

1.  Loads the raw data.
2.  Cleans data (handles missing values, drops irrelevant columns).
3.  Extracts features from timestamps and location names (city, state).
4.  Aggregates data by `trip_uuid` to represent complete trips.
5.  Calculates overall trip duration.
6.  Compares various time/distance metrics visually and statistically.
7.  Identifies and caps outliers using the IQR method.
8.  Encodes categorical features (one-hot encoding).
9.  Standardizes numerical features using `StandardScaler`.
10. Saves the final processed data.
11. Prints basic business insights (top routes, etc.).

Refer to `documentation.md` for a more detailed explanation of each step and the findings.
