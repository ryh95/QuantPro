"""
this module get all hs300 stocks data from tusare
save them into MongoDB

"""
import time
from pymongo import MongoClient
import tushare as ts
import pandas as pd

# read in all symbols
client = MongoClient()
db = client['stockcodes']
coll = db.HS300.find()

# set crawling date
start,end= '2013-10-23','2016-10-19'

db_data = client['data']
missing_stocks = 0
t0 = time.clock()

for document in coll:
    stockcode = document['stockcode']
    df = ts.get_hist_data(stockcode, start, end)
    df.index = pd.to_datetime(df.index)
    # missing data handling
    # details in pandas_missing_data_handling.ipynb
    idx = pd.date_range(start=start, end=end)
    try:
        data = df.reindex(idx)
        zvalues = data.loc[~(data.volume > 0)].loc[:, ['volume', 'amount']]
        data.update(zvalues.fillna(0))
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)
        # iter all rows in data(dataframe) to save them into MongoDB
        for index,row in data.iterrows():
            db_data[stockcode].insert_one(
                {
                    "date":index,
                    "open":row[0],
                    "high":row[1],
                    "close":row[2],
                    "low":row[3],
                    "volume":row[4]
                }
            )
        print '{} has finished!'.format(stockcode)
    except AttributeError,e:
        missing_stocks += 1
        print '{} dataframe is missing and skipped!'.format(stockcode)
t = time.clock()
print '{} stocks are missing , {} cost'.format(missing_stocks,t-t0)