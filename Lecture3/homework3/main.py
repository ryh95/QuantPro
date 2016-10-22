"""
this module checks if there is stock time series  in HS300 is stationary
using ADF and Hurst Exponent method
"""

# read time series from MongoDB
import time
from pymongo import MongoClient
import numpy as np
import statsmodels.tsa.stattools as ts

from Lecture3.homework3.tools import hurst

client = MongoClient()
db = client['stockcodes']
coll = db.HS300.find()


symbol_dict = {}

for document in coll:
    symbol_dict[document['stockcode']] = document['name']


symbols, names = np.array(list(symbol_dict.items())).T


db = client['data']
t0 = time.clock()

# num to record the mean-reverting stocks
num=0

for symbol in symbols:
    # handle time series of one symbol
    coll =  db[symbol].find()
    close_symbol = []
    for document in coll :
        close_symbol.append(document['close'])
    close_symbol = np.array(close_symbol)
    # core algorithm:
    # if time series passses the ADF and Hurst Exponent test
    # then it is mean-reverting
    result_ADF = ts.adfuller(close_symbol,1)
    result_Hurst = hurst(close_symbol)
    if (not result_ADF[0]>result_ADF[4]['5%']) and result_Hurst < 0.5:
        print '{} is mean-reverting !'.format(symbol)
        num += 1

t = time.clock()
print '{} stocks are mean-reverting, {} cost'.format(num,t-t0)