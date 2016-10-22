
from numpy import std, subtract, polyfit, sqrt, log


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # create the range of lag values
    lags = range(2,100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:],ts[:-lag]))) for lag in lags]

    # use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags),log(tau),1)

    # Return the Hurst Exponent from the polyfit output
    return poly[0]*2.0

