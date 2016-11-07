import datetime
import pandas_datareader.data as web
import pandas as pd

start = datetime.datetime(1990,1,1,0,0,0)
end = datetime.datetime(2016,11,7)
pd = web.DataReader("AAPL", "yahoo", start, end)
pd.to_csv('AAPL.csv')