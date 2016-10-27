import pandas as pd


# Compute the Bollinger Bands
def BBANDS(data, ndays):
    MA = pd.Series(pd.rolling_mean(data['Close'], ndays))
    SD = pd.Series(pd.rolling_std(data['Close'], ndays))

    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name='Upper BollingerBand')
    data = data.join(B1)

    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name='Lower BollingerBand')
    data = data.join(B2)

    return data


def SMA(data, ndays):
    '''
     # Simple Moving Average
     :param data: price (pandas series)
     :param ndays: num of days to look back
     :return: simple moving average (pandas series)
    '''
    SMA = pd.Series(pd.rolling_mean(data['Close'], ndays), name='SMA_' + str(ndays))
    data = data.join(SMA)
    return data


def EWMA(data, ndays):
    '''
     # Exponentially-weighted Moving Average
     :param data: price (pandas series)
     :param ndays: num of days to look back
     :return:
     '''
    EMA = pd.Series(pd.ewma(data['Close'], span=ndays, min_periods=ndays - 1), name='EWMA_' + str(ndays))
    data = data.join(EMA)
    return data


def ROC(data, n):
    '''
     # Rate of Change (ROC)
     :param data: price (pandas series)
     :param n: periods to look back
     :return: pandas series
    '''
    # formula
    # ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N / D, name='Rate of Change')
    data = data.join(ROC)
    return data


# Commodity Channel Index
def CCI(data, ndays):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),
                    name='CCI')
    data = data.join(CCI)
    return data


# Force Index
def ForceIndex(data, ndays):
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name='ForceIndex')
    data = data.join(FI)
    return data


# Ease of Movement
def EVM(data, ndays):
    dm = ((data['High'] + data['Low']) / 2) - ((data['High'].shift(1) + data['Low'].shift(1)) / 2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br
    EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name='EVM')
    data = data.join(EVM_MA)
    return data
