import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

tickers = ['INTC'] #, 'MSFT', 'AAPL', 'GOOGL', 'AMZN']

# scrape_url = "https://www.gurufocus.com/stock/INTC/financials"

not_found_tickers = set()


driver = webdriver.Edge()


def process_company(ticker):
    scrape_url = f"https://www.gurufocus.com/stock/{ticker}/financials"

    driver.get(scrape_url)
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.ID, 'per_share_data')))

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # element = soup.find(id='per_share_data')
    table = soup.find_all('table')
    df = pd.read_html(str(table))[0]
    if len(table) > 11:
        print(f"Data complete for {ticker}")

    else:
        print(f"Data not found for {ticker}")
        not_found_tickers.add(ticker)

    df = pd.read_html(str(table))[0]
    df = df.set_index('Year')


try:
    for ticker in tickers:
        try:
            process_company(ticker)

        except TimeoutException as e:
            print(f"TimeoutException occurred for {ticker}")
            not_found_tickers.add(ticker)
        except WebDriverException as e:
            print(f"WebDriverException occurred for {ticker}")
            not_found_tickers.add(ticker)
        except Exception as e:
            print(f"Exception occurred for {ticker}")
            not_found_tickers.add(ticker)

finally:
    driver.quit()
