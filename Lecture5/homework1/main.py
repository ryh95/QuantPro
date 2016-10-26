import tushare as  ts
import pandas as pd

# set crawling date
start,end= '2013-10-23','2016-10-19'
# crawl data
df = ts.get_hist_data(code='hs300',start=start,end=end)
df.to_csv('hs300.csv')

