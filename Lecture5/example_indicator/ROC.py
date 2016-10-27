# Rate of Change code

# Load the necessary packages and modules
import pandas as pd
# import pandas.io.data as web
from pandas_datareader import data as  web
import matplotlib.pyplot as plt
from Lecture5.technique_indicators.indicators import ROC

# Retrieve the NIFTY data from Yahoo finance:
data = web.DataReader('^NSEI',data_source='yahoo',start='6/1/2015',end='1/1/2016')
data = pd.DataFrame(data)

# Compute the 5-period Rate of Change for NIFTY
n = 5
NIFTY_ROC = ROC(data,n)
ROC = NIFTY_ROC['Rate of Change']

# Plotting the Price Series chart and the Ease Of Movement below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(ROC,'k',lw=0.75,linestyle='-',label='ROC')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('ROC values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()