# BITCOIN-Forecast
Bitcoin Closing Price Forecast using LSTM

The Project is a Univariate Time Series Forecast considering the Closing Price As the Values.

![btcpricepredimage](https://github.com/user-attachments/assets/b24a3862-a412-4085-a01b-682c27f9e23a)

Dataset(.csv) is Downloaded from CoinMarketCap = https://coinmarketcap.com/currencies/bitcoin/
The Dataset Gives Monthly Data from 2013-04-01 until 2024-09-01

# ETL
Firstly i Reformat the Date,Clean,and Normalized the Data;
Set the Sequence and Labels (i set the Len Sequence to 30);
Split the Training And Testing (80% and 20%);

# MODEL
For the layer(s) I use LSTM Because the Data is not Stationary,and End with Dense layer;
Compile And Fit the Model,but dont forget to reshape the Input Data to align with LSTM input shape (sample,lensequence,feature);

# PLOT
Predict the Model and Inverse Normalized the Result;
Plot the Actual Values And the Result Values.
