# TSA_UK_Electricity
## Group:
* Zihang WANG
* Liyang FANG
* Sanjaya deshapriya gunawardena Heeralu aarachchige don

## DataSet Info:
### Info
National Grid ESO is the electricity system operator for Great Britain. They have gathered information of the electricity demand in Great Britain from 2009. The is updated twice an hour, which means 48 entries per day. This makes this dataset ideal for time series forecasting.

The [dataset](https://www.kaggle.com/datasets/albertovidalrod/electricity-consumption-uk-20092022) can be downloaded from Kaggle, we choose the historic_demand_2009_2024_noNaN.csv for the next analysis.

### Feature
The main features we will use: 

* SETTLEMET_DATA: date in format dd/mm/yyyy
* SETTLEMENT_PERIOD: half hourly period for the historic outtunr occurred
* TSD (Transmission System Demand): Transmission System Demand is equal to the ND plus the additional generation required to meet station load, pump storage pumping and interconnector exports. Measured in MW.

### Quick look:
* The dataset has high seasonality
* Using a simple statistics tool-- Z score, it shows several anomalies.
* For more details, please checkt the notebook.   
