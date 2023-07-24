import requests
from bs4 import BeautifulSoup

def get_altman_z_score(ticker):
    url = f"https://www.gurufocus.com/stock/{ticker}/summary?search="
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the Altman Z-score in the page
    altman_z_score = None
    for div in soup.find_all('div', {'class': 'data_box'}):
        if 'Altman Z-Score' in div.text:
            altman_z_score = div.find('span', {'class': 'data_box_value'}).text
            break

    if altman_z_score is None:
        print(f"Altman Z-Score not found for ticker {ticker}")
    else:
        print(f"Altman Z-Score for {ticker} is {altman_z_score}")

# Test the function
get_altman_z_score('MSFT')
