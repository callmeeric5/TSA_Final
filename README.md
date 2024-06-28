# KPI_Anomaly_Detection
## Group:
* Zihang WANG
* Liyang FANG

## DataSet Info:
### Info
With the rapid development of the Internet, especially the mobile Internet, web services have penetrated into all areas of society. People use the Internet to search, shop, pay, entertain, etc. Therefore, ensuring the stability of web services has become increasingly important.
The stability of web services is mainly guaranteed by operation and maintenance. Operation and maintenance personnel judge whether the web service is stable by monitoring various key performance indicators (KPIs), because if a KPI is abnormal, it often means that the application related to it has failed. question.

The [data](https://competition.aiops-challenge.com/home/competition/1484452272200032281#1501482986716332093index=0) is from AIOps competation and it can be downloaded from [here](https://github.com/NetManAIOps/KPI-Anomaly-Detection/tree/master/Preliminary_dataset).

### Feature
The features are: 

* KPI ID
* timestamp
* value 
* tag
  
### Quick look:
* The dataset is complicated containing 26 different kinds of time series data.
* For more details, please checkt the notebook.   

### Check data info
* Before finding the anomalies, check whether the dataset has missing values (each KPI ID)
* Filling the missing values
* Visualizing the anomalies and their distribution

### Data preprocessing
* According to the visualization, we chose three KPI ID to represent each situation which are period, stable and unstable. 


### Data visulization
* Staionarity
* Seasonality
* ACF&PACF

### Anomaly detection
* Heuristics based STL Decomposition
* GMM
* LSTM Autoencoder

### Model comparison
![39dbfe916d81c5c61de6f3465c00de6](https://github.com/Icybrig/recsys/assets/136721036/633141e4-85c3-4d76-bade-8a092e44af62)

* We have used three models (STL, GMM, LSTM Autoencoder) in three KPI ID. For the accuracy, from the graph we can see that LSTM Autoencoder is the worst one for all. STL and GMM are both good, but STL is better than GMM in two KPI ID. Therefore, we could say STL is the best one. 
* If we combining all the values, STL performs well generally but struggle with low recall in some cases such as the third KPI ID. GMM performs well with good balance of precision and recall for all KPI. LSTM Autoencoder shows high in recall so it is useful for identify true positive even it has lower accuracy. 

# model
* Seasonal and Trend Decomposition using Loess (STL) : To decompose the series using seasonal decomposition. Then calculate z-scores using either median and MAD to identify anomalies where absolute z-scores exceed the threshold. Finally, evaluate the detected anomalies against ground truth and plot results.
* Gaussian Mixture Model (GMM) Detection : Preparing and scaling the data. To fit a GMM to the scaled data. Then calculate data point densities for identify anomalies based on a percentile threshold.Evaluating and plots the detected anomalies.
* Recurrent Neural Network Autoencoder (LSTM Autoencoder) :
    Encoder : Two LSTM layers are used to reduce the dimensionality and capture the sequence information.
    Decoder : Two LSTM layers are used to reconstruct the sequences followed by a linear layer to output the original dimensions.
    RecurrentAutoencoder : Connecting Encoder and Decoder to run from input to output to construct the original data. 

# utilities
* Plotting Anomalies with Ground Truth Labels : plot along with anomalies (label=1) marked as red dots
* Plotting Detected Anomalies : Add the detected anomalies marked as red marker.
* Stationarity Test : To use Dickey-Fuller test to check the stationarity and plot the Original Time Series, Rolling Mean and Rolling Std. Determine its stationarity on the test statistics and p-value.
* Seasonal Decomposition : Separating the time series into trend, seasonal, and residual components. To plot it as visualization.
* ACF and PACF Plot : ACF and PACF plots for time series to identify the correlation between the current and lagged values of the series.
* Evaluation Metrics : To calculate and make a dataframe with accuracy, F1, Precision, Recall, FPR and FNR.
* Normalize Time Series Dataset : Using Pytorch and StandardScalar to normalize time series data into the sequences of a specified length.
* Data Loader Function : Batching and shuffling time series data sequences