import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
)


def plot_anomaly_labeled(df):
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


def plot_anomaly_detected(df, anomalies):
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=df, x="datetime", y="value")
    sns.scatterplot(
        data=df[df.label == 1],
        x="datetime",
        y="value",
        color="g",
        s=50,
        alpha=0.7,
        label="anomaly (label=1)",
    )
    sns.scatterplot(
        data=anomalies.value,
        x=anomalies["datetime"],
        y=anomalies["value"],
        color="r",
        s=100,
        alpha=0.7,
        label="anomaly (detected)",
        marker="X",
    )
    plt.title(f"Anomalies for KPI ID {df.kpi_id.iloc[0]}")
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
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


def season_decompose(df, isplot=True):
    df["value"] = df["value"].interpolate()
    decomposition = seasonal_decompose(df["value"], model="additive", period=1440)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    if isplot:
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


def evaluation(y_true, y_pred, kpi_id, method):
    # Confusion matrix
    print("Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    print(f"True Positive: {TP}")
    print(f"False Positive: {FP}")
    print(f"False Negative: {FN}")
    print(f"True Negative: {TN}")

    # False positive rate
    FPR = round(FP / (FP + TN) * 100, 2)
    print(f"False Positive Rate: {FPR}%")

    # False negative rate
    FNR = round(FN / (TP + FN) * 100, 2) if FN > 0 else 0

    print(f"False Negative Rate: {FNR}%")

    # Accuracy
    ACC = round((TP + TN) / (TP + FP + FN + TN) * 100, 2)
    print("Accuracy: {:.2f}%".format(ACC))
    F1 = round(f1_score(y_true, y_pred) * 100, 2)
    print("F1 Score: {:.2f}%".format(F1))
    Recall = round(recall_score(y_true, y_pred) * 100, 2)
    print("Recall: {:.2f}%".format(Recall))
    Precision = round(precision_score(y_true, y_pred) * 100, 2)
    print("Precision: {:.2f}%".format(Precision))

    columns = [
        "kpi_id",
        "method",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "FPR",
        "FNR",
    ]
    eval_df = pd.DataFrame(
        [[kpi_id, method, ACC, F1, Precision, Recall, FPR, FNR]],
        columns=columns,
    )
    return eval_df

class NormalTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len):
        self.data = df['value'].values.reshape(-1, 1)  # Reshape to (num_samples, 1)
        self.seq_len = seq_len
        
        # Normalize the data
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, index):
        return torch.FloatTensor(self.normalized_data[index:index+self.seq_len])

def data_loader(normal_data, seq_len, batch_size):
    dataset = NormalTimeSeriesDataset(normal_data, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)