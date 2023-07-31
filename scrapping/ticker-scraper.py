import pandas as pd
from yahoo_fin import stock_info as si
import json

# Get ticker symbols
# df2 = pd.DataFrame(si.tickers_nasdaq())
#
# # Convert to set and then list
# sym2 = list(set(symbol for symbol in df2[0].values.tolist()))
#
# # Print the symbols
# print(sym2)
#
# # Write to JSON file
# with open('tickers.json', 'w') as f:
#     json.dump(sym2, f)

nasdaq_tickers = []

# Open the file and read each line
with open('NASDAQ.txt', 'r') as f:
    next(f)  # Skip the header line
    for line in f:
        symbol = line.split('\t')[0]
        nasdaq_tickers.append(symbol)

print(f'Number of NASDAQ tickers: {len(nasdaq_tickers)}')

my_list = ['W', 'R', 'P']

# W means there are outstanding warrants. We don’t want those.
# R means there is some kind of “rights” issue. Again, not wanted.
# P means “First Preferred Issue”. Preferred stocks are a separate entity.
# Q means bankruptcy. We want those.

sav_set = set()

for symbol in nasdaq_tickers:
    if len(symbol) > 4 and symbol[-1] in my_list:
        continue
    else:
        sav_set.add(symbol)

print(f'There are {len( sav_set )} qualified stock symbols...')

with open('qualified-tickers.json', 'w') as f:
    json.dump(list(sav_set), f)

print(sav_set)
