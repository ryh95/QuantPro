from datetime import datetime

import numpy as np
import statsmodels.tsa.stattools as ts
import pandas_datareader.data as web

amzn = web.DataReader("AMZN","yahoo",datetime(2000,1,1),datetime(2015,1,1))
print ts.adfuller(amzn['Adj Close'],1)