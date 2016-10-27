import pandas as pd


# Compute the Bollinger Bands
def BBANDS(data, ndays,close='Close'):
    MA = pd.Series(pd.rolling_mean(data[close], ndays))
    SD = pd.Series(pd.rolling_std(data[close], ndays))

    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name='Upper BollingerBand')
    data = data.join(B1)

    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name='Lower BollingerBand')
    data = data.join(B2)

    return data


def SMA(data, ndays,close='Close'):
    '''
     # Simple Moving Average
     :param data: price (pandas series)
     :param ndays: num of days to look back
     :return: simple moving average (pandas series)
    '''
    SMA = pd.Series(pd.rolling_mean(data[close], ndays), name='SMA_' + str(ndays))
    data = data.join(SMA)
    return data


def EWMA(data, ndays,close='Close'):
    '''
     # Exponentially-weighted Moving Average
     :param data: price (pandas series)
     :param ndays: num of days to look back
     :return:
     '''
    EMA = pd.Series(pd.ewma(data[close], span=ndays, min_periods=ndays - 1), name='EWMA_' + str(ndays))
    data = data.join(EMA)
    return data


def ROC(data, n,close='Close'):
    '''
     # Rate of Change (ROC)
     :param data: price (pandas series)
     :param n: periods to look back
     :return: pandas series
    '''
    # formula
    # ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    N = data[close].diff(n)
    D = data[close].shift(n)
    ROC = pd.Series(N / D, name='Rate of Change')
    data = data.join(ROC)
    return data


# Commodity Channel Index
def CCI(data, ndays,high='High',low='Low',close='Close'):
    TP = (data[high] + data[low] + data[close]) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),
                    name='CCI')
    data = data.join(CCI)
    return data


# Force Index
def ForceIndex(data, ndays,close='Close',volume='Volume'):
    FI = pd.Series(data[close].diff(ndays) * data[volume], name='ForceIndex')
    data = data.join(FI)
    return data


# Ease of Movement
def EVM(data, ndays,high='High',low='Low',volume='Volume'):
    dm = ((data[high] + data[low]) / 2) - ((data[high].shift(1) + data[low].shift(1)) / 2)
    br = (data[volume] / 100000000) / ((data[high] - data[low]))
    EVM = dm / br
    EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name='EVM')
    data = data.join(EVM_MA)
    return data
