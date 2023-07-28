from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import glob
import time
import datetime
import tqdm

login_url = "https://ycharts.com/login"
# scrape_url = "https://ycharts.com/companies/WMT/altman_z_score"

payload = {
    "username": "darekkruszel15@gmail.com",
    "password": "!Jaszczur15!"
}

df = pd.DataFrame(columns=['AltmanZScore'])
not_found_tickers = []

driver = webdriver.Chrome()

cnt = 0
files_count = 0

try:
    driver.implicitly_wait(5)

    driver.get(login_url)

    username = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'username')))
    password = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'password')))

    # Don't hardcode sensitive information like usernames and passwords.
    # Instead, consider using input prompts or environment variables.
    username.clear()
    password.clear()
    username.send_keys(payload['username'])
    password.send_keys(payload['password'])

    password.send_keys(Keys.RETURN)

    # file things
    directory = '../dataset/altman-z-score/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    json_file_path = 'tickers.json'

    with open(json_file_path, 'r') as file:
        tickers = json.load(file)

    files = [f for f in glob.glob(f"{directory}*") if os.path.isfile(f)]

    # Count the number of files
    files_count = len(files)

    json_file_path = 'not_found_tickers_2023-07-28_21-13-34.json'

    with open(json_file_path, 'r') as file:
        not_found_tickers = json.load(file)

    beginning_index = files_count + len(not_found_tickers)

    for ticker in tickers[beginning_index:]:
        scrape_url = f"https://ycharts.com/companies/{ticker}/altman_z_score"

        driver.get(scrape_url)

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        h1_tag = soup.find('h1', string='Page Not Found')

        if h1_tag or driver.current_url != scrape_url:
            not_found_tickers.append(ticker)
            continue
        else:
            page_source = driver.page_source

            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'tr')))

            page_source = driver.page_source

            soup = BeautifulSoup(page_source, 'html.parser')

            custom_element = soup.find('ycn-historical-data-table')
            table = custom_element.find('div', {'class': 'row'})

            for row in table.find_all('tr'):
                row = [cell.text for cell in row.find_all('td')]
                if len(row) == 0:
                    continue

                try:
                    date = pd.to_datetime(row[0])
                except ValueError:
                    date = None

                try:
                    score = float(row[1].strip())
                except ValueError:
                    score = None

                df.loc[date] = score

            df.index.name = 'DateTime'

            df.to_csv(f'{directory}{ticker}.csv')

            cnt += 1
            time.sleep(1)

except Exception as e:
    print(f"Ticker counter: {cnt + files_count}")
    print(f"An error occurred: {e}")
finally:
    now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(f'not_found_tickers_{now_str}.json', 'w') as f:
        json.dump(not_found_tickers, f)
    driver.quit()
