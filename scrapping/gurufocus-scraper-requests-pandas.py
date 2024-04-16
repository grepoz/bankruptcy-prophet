import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from collections import ChainMap

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'authority': 'www.google.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    # Add more headers as needed
}

url = 'https://www.gurufocus.com/stock/INTC/financials'

response = requests.get(url, headers=headers)
response_text = response.text

tables = pd.read_html(
    response_text,
    header=0
)
print(tables)
sub_table_values = [[{record["Name"]: record["Current"]} for record in json.loads(e)] for e in [i.to_json(orient="records") for i in tables]]
sub_formatted = [dict(ChainMap(*a)) for a in sub_table_values]
print(json.dumps(sub_formatted, indent=4))
