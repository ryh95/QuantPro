import re
from pymongo import *
# import pymongo

# this script is used to insert the HS300 stockcodes into MongoDB
client = MongoClient()
db = client['stockcodes']

f= open('HS300.txt','r')
all_text = f.read()
stocks = all_text.split('\r\n')
for stock in stocks:
    result = db.HS300.insert_one({
        "stockcode" : stock[:6],
        "name":stock.split('\t')[1]
    })
    print result.inserted_id
f.close()