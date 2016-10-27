################# Force Index ########################################################

# Load the necessary packages and modules
import pandas as pd
from pandas_datareader import data as web

from Lecture5.tools.indicators import ForceIndex

data = web.DataReader('AAPL',data_source='yahoo',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data)

# Compute the Force Index for Apple 
n = 1
AAPL_ForceIndex = ForceIndex(data,n)
print(AAPL_ForceIndex)