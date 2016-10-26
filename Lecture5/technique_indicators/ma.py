import pandas as pd

# Simple Moving Average
def SMA(data, ndays):
 SMA = pd.Series(pd.rolling_mean(data['Close'], ndays), name = 'SMA_'+str(ndays))
 data = data.join(SMA)
 return data

# Exponentially-weighted Moving Average
def EWMA(data, ndays):
 EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1), name = 'EWMA_' + str(ndays))
 data = data.join(EMA)
 return data
