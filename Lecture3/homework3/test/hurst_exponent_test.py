from datetime import datetime


from numpy import cumsum, log
from numpy.random.mtrand import randn

from Lecture3.homework3.tools import hurst
import pandas_datareader.data as web

# creating a Gometric Brownian Motion , Mean-Reverting and Trending Series

gbm = log(cumsum(randn(100000))+1000)
mr = log(randn(100000)+1000)
tr = log(cumsum(randn(100000)+1)+1000)

# output the Hurst Exponent for each of the above series
# and the price of Amazon (the Adjusted Close price) for
# the ADF test given in the other test
print 'Hurst(GBM):  %s' % hurst(gbm)
print 'Hurst(MR):  %s' % hurst(mr)
print 'Hurst(TR):  %s' % hurst(tr)

amzn = web.DataReader("AMZN","yahoo",datetime(2000,1,1),datetime(2015,1,1))
print 'Hurst(AMAZON):  %s' % hurst(amzn['Adj Close'])