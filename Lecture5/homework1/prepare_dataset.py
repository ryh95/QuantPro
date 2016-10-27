import numpy as  np
import pandas as pd

from Lecture5.tools.indicators import EWMA ,ROC, CCI,ForceIndex

df = pd.read_csv('hs300.csv')

# # create features
def create_features(df,lags=5):
    X = df.set_index(df['date'])
    X.sort_values(by='date',inplace=True)
    X.drop('date',axis=1,inplace=True)
    X.index = pd.to_datetime(X.index)

    # add lags
    for i in xrange(0, lags):
        X["lag%s" % str(i+1)] = X["close"].shift(i+1)

    # add return
    X['return'] = X['close'].pct_change()*100.0
    for i in xrange(0,lags):
        X['return'+str(i+1)] = X['lag'+str(i+1)].pct_change()*100.0

    # add indicators
    X = EWMA(X,lags,close='close')
    X = ROC(X,lags,close='close')
    X = CCI(X,lags,high='high',low='low',close='close')
    X = ForceIndex(X,lags,close='close',volume='volume')

    # get rid of rows which contains nan values
    X.dropna(inplace=True)

    return X

df = create_features(df)

# feature rescale
df_norm = (df - df.mean()) / (df.max() - df.min())

# add direction as target value
df_norm['direction'] = np.sign(df['return'])

df_norm.to_csv('hs300_dataset.csv')