from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd

login_url = "https://ycharts.com/login"
scrape_url = "https://ycharts.com/companies/WMT/altman_z_score"

payload = {
    "username": "darekkruszel15@gmail.com",
    "password": "set-env-var"
}

df = pd.DataFrame(columns=['altman z-score'])

driver = webdriver.Chrome()

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

    print("Login successful")

    driver.get(scrape_url)
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'tr')))

    a = driver.page_source

    soup = BeautifulSoup(a, 'html.parser')

    custom_element = soup.find('ycn-historical-data-table')
    table = custom_element.find('div', {'class': 'row'})

    for row in table.find_all('tr'):
        row = [cell.text for cell in row.find_all('td')]
        if len(row) == 0:
            continue
        date = pd.to_datetime(row[0])
        score = float(row[1].strip())

        df.loc[date] = score

    print(df)

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    driver.quit()
