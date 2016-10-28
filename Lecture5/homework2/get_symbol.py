import time
from pymongo import MongoClient
import tushare as ts
import pandas as pd
import numpy as np

client = MongoClient()
db = client['stockcodes']
coll = db.HS300.find()

t0 = time.clock()

start,end= '2013-10-23','2016-10-19'

stocks_condition1 = []
ADV = []

stockcodes = [document['stockcode'] for document in coll]

for stockcode in stockcodes:
    # stockcode = document['stockcode']
    df = ts.get_hist_data(stockcode, start, end)
    df.index = pd.to_datetime(df.index)
    if len(df.query('close<10 | close>50')) == 0:
        stocks_condition1.append(stockcode)
    ADV.append(df['volume'].mean())

df = pd.DataFrame(index=stockcodes)
df['ADV'] = np.array(ADV)

# df = pd.DataFrame(index=range(1,11))
# df['ADV'] = np.array(range(1,11))

stocks_condition3 =  df[((df-df.min())/(df.max()-df.min())>0.33) & ((df-df.min())/(df.max()-df.min())<0.66)].index.tolist()

print list(set(stocks_condition1)&set(stocks_condition3))
print time.clock()-t0