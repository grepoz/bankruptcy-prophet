import json
import os
import glob

if __name__ == '__main__':

    directory = 'dataset/altman-z-score-incorrect/'

    if not os.path.exists(directory):
        print('directory does not exist')

    tickers_json_file_path = 'scrapping/tickers.json'

    with open(tickers_json_file_path, 'r') as file:
        tickers = json.load(file)

    files = [f for f in glob.glob(f"{directory}*") if os.path.isfile(f)]

    # Count the number of files
    files_count = len(files)

    not_found_tickers_json_file_path = 'scrapping/not_found_tickers_2023-07-28_23-29-05.json'

    with open(not_found_tickers_json_file_path, 'r') as file:
        not_found_tickers = json.load(file)

    print(f'Number of tickers: {len(tickers)}, not found tickers: {len(not_found_tickers)}, files: {files_count}')
