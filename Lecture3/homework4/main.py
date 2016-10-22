
# get data first
import time
from pymongo import MongoClient
import numpy as np
import pandas as pd
from pandas.stats.api import ols
import statsmodels.tsa.stattools as ts

client = MongoClient()
db = client['stockcodes']
coll = db.HS300.find()


symbol_dict = {}

for document in coll:
    symbol_dict[document['stockcode']] = document['name']


symbols, names = np.array(list(symbol_dict.items())).T


db_data = client['data']

# create a data frame
start = '20131023'
dates = pd.date_range(start,periods=1093)
df = pd.DataFrame(index=dates)

for symbol in symbols:
    coll =  db_data[symbol].find()
    close_symbol = []
    for document in coll:
        close_symbol.append(document['close'])
    close_symbol = np.array(close_symbol)
    df[symbol] = close_symbol
    # print type(symbol)

# df.to_csv('test.csv')

# close = np.array(close).astype(np.float)

# get headers
headers = list(df.columns.values)

# core algorithm below
t0 = time.clock()
num_pairs = 0
for i in xrange(299):
    for j in xrange(i,300):
        Y = df.iloc[:,j]
        X = df.iloc[:,i]
        # linear fit
        res = ols(y=Y, x=X)
        beta_hr = res.beta.x
        # get residual
        residual = Y - beta_hr*X
        # check residual stationary or not
        result_ADF = ts.adfuller(residual)
        if result_ADF[0]<result_ADF[4]['5%']:
            print '{} and {} is a pair'.format(headers[i],headers[j])
            num_pairs += 1
t = time.clock()
print 'total {} pairs , {} cost'.format(num_pairs,t-t0)
