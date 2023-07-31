import datetime
import json
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
import traceback

login_url = "https://ycharts.com/login"
# scrape_url = "https://ycharts.com/companies/SSSSL/altman_z_score"

payload = {
    "username": "kruszel15@gmail.com",
    "password": "!Jaszczur15!"
}

not_found_tickers = set()

driver = webdriver.Chrome()

iter_counter = 0
cnt = 0
files_count = 0
ticker = ''

try:
    driver.get(login_url)

    username = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'username')))
    password = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'password')))
    username.clear()
    password.clear()
    username.send_keys(payload['username'])
    password.send_keys(payload['password'])
    password.send_keys(Keys.RETURN)

    # file things
    directory = Path('../dataset/altman-z-score/')
    directory.mkdir(parents=True, exist_ok=True)

    tickers = json.loads(Path('qualified-tickers.json').read_text())

    files_count = len(list(directory.glob('*')))

    not_found_tickers = set(json.loads(Path('not_found_tickers_2023-07-29_11-32-36.json').read_text()))

    beginning_index = files_count + len(not_found_tickers) - 1

    for ticker in tqdm(tickers[beginning_index:]):

        scrape_url = f"https://ycharts.com/companies/{ticker}/altman_z_score"

        driver.get(scrape_url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        if soup.find('h1', string='Page Not Found') or driver.current_url != scrape_url:
            not_found_tickers.add(ticker)
            continue

        try:
            page_source = driver.page_source

            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'tr')))

            soup = BeautifulSoup(page_source, 'html.parser')

            df = pd.DataFrame(columns=['AltmanZScore'])

            custom_element = soup.find('ycn-historical-data-table')
            table = custom_element.find('div', {'class': 'row'})

            for row in table.find_all('tr'):
                cells = [cell.text for cell in row.find_all('td')]
                if len(cells) == 0:
                    continue

                try:
                    date = pd.to_datetime(cells[0])
                except ValueError:
                    date = None

                try:
                    score = float(cells[1].strip())
                except ValueError:
                    score = None

                df.loc[date] = score

            df.index.name = 'DateTime'
            df.to_csv(directory / f'{ticker}.csv')
            cnt += 1
            time.sleep(0.5)

        except (TimeoutException, ValueError, WebDriverException) as e:
            print(f"An error occurred while processing {ticker}: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error trace: {traceback.format_exc()}")
            print(f"Current URL: {driver.current_url}")
            print(f"Page title: {driver.title}")
            continue

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error trace: {traceback.format_exc()}")
    print(f"Current URL: {driver.current_url}")
    print(f"Page title: {driver.title}")
finally:
    print(f"Iter counter: {iter_counter}, Ticker counter: {cnt + files_count}, ticker: {ticker})")
    now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    Path(f'not_found_tickers_{now_str}.json').write_text(json.dumps(not_found_tickers))
    driver.quit()
