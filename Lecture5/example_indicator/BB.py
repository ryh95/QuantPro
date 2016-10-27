################ Bollinger Bands #############################

# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data as web
 
# Retrieve the Nifty data from Yahoo finance:
from Lecture5.technique_indicators.indicators import BBANDS

data = web.DataReader('^NSEI',data_source='yahoo',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data)

# Compute the Bollinger Bands for NIFTY using the 50-day Moving average
n = 50
NIFTY_BBANDS = BBANDS(data, n)
print(NIFTY_BBANDS)