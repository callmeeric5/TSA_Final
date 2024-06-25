import pandas as pd
import numpy as np
from utilities import (
    evaluation,
    season_decompose,
    plot_anomaly_detected,
    plot_anomaly_labeled,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


def STL_decompose(df, threshold=2.5, window_size=1440, use_median=True):

    trend, seasonal, residual = season_decompose(df, isplot=False)

    if use_median:
        # Use median and Median Absolute Deviation (MAD)
        center = residual.rolling(window=window_size).median()
        mad = residual.rolling(window=window_size).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        z_scores = 0.6745 * (residual - center) / mad
    else:
        # Use mean and standard deviation
        center = residual.rolling(window=window_size).mean()
        std = residual.rolling(window=window_size).std()
        z_scores = (residual - center) / std

    threshold = threshold * np.ones_like(z_scores)

    # Identify anomalies
    anomalies = df[np.abs(z_scores) > threshold]
    anomalies["score"] = z_scores[np.abs(z_scores) > threshold]

    df["predicted"] = 0
    df.loc[anomalies.index, "predicted"] = 1

    # Print evaluation metrics
    eval_df = evaluation(
        df["label"], df["predicted"], kpi_id=df.kpi_id.iloc[0], method="STL"
    )

    # Plot the results
    plot_anomaly_detected(df.reset_index(), anomalies.reset_index())

    return anomalies, eval_df


def GMM_detection(
    df,
    n_components=3,
    sensity=0.01,
    feature_columns=["value"],
):
    # Prepare the feature matrix
    X = df[feature_columns].copy()

    X = X.fillna(method="ffill").fillna(
        method="bfill"
    )  # Forward fill, then backward fill any remaining NaNs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)

    # Compute the density for each point
    densities = gmm.score_samples(X_scaled)
    # print(densities)
    # Identify anomalies
    threshold = np.percentile(densities, sensity * 100)
    anomalies = df.loc[X.index][densities < threshold]
    df["predicted"] = 0
    df.loc[anomalies.index, "predicted"] = 1

    # Print evaluation metrics
    eval_df = evaluation(
        df["label"], df["predicted"], kpi_id=df.kpi_id.iloc[0], method="GMM"
    )

    # Plot the results
    plot_anomaly_detected(df.reset_index(), anomalies.reset_index())

    return anomalies, eval_df


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.squeeze(0)  # Squeeze to remove the batch_size dimension

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # Expand dimensions to match (batch_size, seq_len, embedding_dim)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = self.output_layer(x)

        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, device, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded