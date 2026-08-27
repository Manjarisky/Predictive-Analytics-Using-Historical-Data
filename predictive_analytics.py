# ============================================================
# PREDICTIVE ANALYTICS USING HISTORICAL DATA
# THIRANEX PROJECT
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. CREATE HISTORICAL SALES DATA
# ============================================================

print("=" * 60)
print("PREDICTIVE ANALYTICS USING HISTORICAL DATA")
print("=" * 60)

np.random.seed(42)

# Generate 48 months of historical sales
dates = pd.date_range(
    start="2022-01-01",
    periods=48,
    freq="MS"
)

months = np.arange(1, 49)

# Create an increasing sales trend with some seasonal variation
sales = (
    50000
    + months * 1200
    + 8000 * np.sin(2 * np.pi * months / 12)
    + np.random.normal(0, 3000, 48)
)

sales = sales.round(2)

df = pd.DataFrame({
    "Date": dates,
    "Sales": sales
})

# Save historical data
df.to_csv(
    "historical_sales.csv",
    index=False
)

print("\nHistorical dataset created successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Sort data by date
df = df.sort_values("Date")

print("\nData cleaning completed.")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Create a numerical time feature
df["Month_Number"] = np.arange(1, len(df) + 1)

# Extract year and month
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

print("\nFeatures created:")
print("- Month_Number")
print("- Year")
print("- Month")


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nStatistical summary:")
print(df["Sales"].describe())


# Historical sales trend
plt.figure(figsize=(10, 6))

plt.plot(
    df["Date"],
    df["Sales"],
    marker="o"
)

plt.title("Historical Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "historical_sales_trend.png"
)

plt.close()

print("\nHistorical trend visualization saved.")


# ============================================================
# 5. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 60)

# Use time as the predictor
X = df[["Month_Number"]]

y = df["Sales"]

# Time-based train-test split
# First 80% = training
# Last 20% = testing

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. TRAIN LINEAR REGRESSION MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print("\nLinear Regression model trained successfully.")

print(
    "Model coefficient:",
    round(model.coef_[0], 2)
)

print(
    "Model intercept:",
    round(model.intercept_, 2)
)


# ============================================================
# 7. MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nTest predictions generated.")


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMean Absolute Error (MAE):")
print(round(mae, 2))

print("\nRoot Mean Squared Error (RMSE):")
print(round(rmse, 2))

print("\nR² Score:")
print(round(r2, 4))


# ============================================================
# 9. ACTUAL VS PREDICTED SALES
# ============================================================

comparison = pd.DataFrame({
    "Date": df["Date"].iloc[split_index:],
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred.round(2)
})

print("\nActual vs Predicted:")
print(comparison)

comparison.to_csv(
    "actual_vs_predicted.csv",
    index=False
)


# ============================================================
# 10. VISUALIZE ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    comparison["Date"],
    comparison["Actual_Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    comparison["Date"],
    comparison["Predicted_Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title(
    "Actual vs Predicted Sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png"
)

plt.close()


# ============================================================
# 11. FUTURE SALES FORECAST
# ============================================================

print("\n" + "=" * 60)
print("FUTURE SALES FORECAST")
print("=" * 60)

# Forecast next 12 months

future_month_numbers = np.arange(
    len(df) + 1,
    len(df) + 13
)

future_dates = pd.date_range(
    start=df["Date"].max()
    + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)

future_predictions = model.predict(
    future_month_numbers.reshape(-1, 1)
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_predictions.round(2)
})

print("\nNext 12 months forecast:")
print(future_df)

future_df.to_csv(
    "future_sales_forecast.csv",
    index=False
)


# ============================================================
# 12. VISUALIZE FUTURE FORECAST
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_df["Date"],
    future_df["Predicted_Sales"],
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

plt.title(
    "Historical Sales and Future Forecast"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "future_sales_forecast.png"
)

plt.close()


# ============================================================
# 13. TREND ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TREND ANALYSIS")
print("=" * 60)

first_sales = df["Sales"].iloc[0]

last_sales = df["Sales"].iloc[-1]

change = last_sales - first_sales

percentage_change = (
    change / first_sales
) * 100

print(
    "\nInitial Sales:",
    round(first_sales, 2)
)

print(
    "Latest Historical Sales:",
    round(last_sales, 2)
)

print(
    "Change:",
    round(change, 2)
)

print(
    "Percentage Change:",
    round(percentage_change, 2),
    "%"
)

if percentage_change > 0:

    print(
        "\nTrend: Overall sales show an increasing trend."
    )

elif percentage_change < 0:

    print(
        "\nTrend: Overall sales show a decreasing trend."
    )

else:

    print(
        "\nTrend: Sales are relatively stable."
    )


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles created:")

print("1. historical_sales.csv")
print("2. actual_vs_predicted.csv")
print("3. future_sales_forecast.csv")
print("4. historical_sales_trend.png")
print("5. actual_vs_predicted.png")
print("6. future_sales_forecast.png")

print("\nModel used:")
print("Linear Regression")

print("\nEvaluation metrics:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 4))

print("\nFuture forecast generated for 12 months.")
