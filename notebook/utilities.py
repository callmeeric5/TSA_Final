import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_anomaly(df):
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=df, x="datetime", y="value")
    sns.scatterplot(
        data=df[df.label == 1],
        x="datetime",
        y="value",
        color="r",
        s=50,
        alpha=0.7,
        label="anomaly (label=1)",
    )
    plt.xlim()
    plt.legend()
    plt.title(f"KPI ID {df.kpi_id.iloc[0]}")  # Display KPI ID in title
    plt.ylabel("Values for KPI ID")
    plt.xlabel("Datetime")
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.show()


def test_stationarity(df):
    df.set_index("datetime", inplace=True)

    # window = 1D => 1440 = 24 * 60
    movingAverage = df["value"].rolling(window=1440).mean()
    movingSTD = df["value"].rolling(window=1440).std()

    # Plot rolling statistics
    plt.figure(figsize=(15, 5))
    plt.plot(df["value"], color="blue", label="Original")
    plt.plot(movingAverage, color="red", label="Rolling Mean")
    plt.plot(movingSTD, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title(f"KPI ID {df.kpi_id.iloc[0]}")
    plt.show()

    # Perform Dickey–Fuller test:
    print("Results of Dickey Fuller Test:")
    kpi_test = adfuller(df["value"], autolag="AIC")
    dfoutput = pd.Series(
        kpi_test[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in kpi_test[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    if (kpi_test[1] < 0.05) and (kpi_test[0] < kpi_test[4]["5%"]):
        print(f"KPI ID {df.kpi_id.iloc[0]} is stationary")
    else:
        print(f"KPI ID {df.kpi_id.iloc[0]} is NOT stationary")


def season_decompose(df):
    decomposition = seasonal_decompose(df["value"], model="additive", period=1440)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    df["value"].plot(ax=axes[0], color="blue", label="Original")
    axes[0].set_title(f"KPI ID {df.kpi_id.iloc[0]}")

    trend.plot(ax=axes[1], color="red", label="Trend")
    axes[1].set_title("Trend")

    seasonal.plot(ax=axes[2], color="green", label="Seasonality")
    axes[2].set_title("Seasonality")

    residual.plot(ax=axes[3], color="purple", label="Residuals")
    axes[3].set_title("Residuals")

    plt.tight_layout()
    plt.show()

    return trend, seasonal, residual


def plot_acf_pacf(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    plot_acf(df["value"], lags=100, ax=ax1)
    ax1.set_title(f"ACF for KPI ID {df.kpi_id.iloc[0]}")

    plot_pacf(df["value"], lags=100, ax=ax2)
    ax2.set_title(f"PACF for KPI ID {df.kpi_id.iloc[0]}")

    plt.show()
