import pandas as pd
import string

def flatten_financial_dataset(financial_dataset):
    object_length_in_rows = 5
    metadata_columns_length = 5

    per_object_columns = list(financial_dataset.columns[:metadata_columns_length].values)

    value_columns = financial_dataset.columns[metadata_columns_length:]
    new_columns = per_object_columns + [f'{col}_{i + 1}' for i in range(object_length_in_rows) for col in value_columns]

    dfs = []

    for i in range(0, len(financial_dataset), object_length_in_rows):
        group = financial_dataset.iloc[i:i + object_length_in_rows]
        if len(group) < object_length_in_rows:
            break

        cik = group['cik'].iloc[0]
        ticker = group['ticker'].iloc[0]
        label = group['label'].iloc[0]
        subset = group['subset'].iloc[0]
        fiscal_periods = ';'.join(group['Fiscal Period'].astype(str).values)

        values = group.drop(columns=per_object_columns).values.flatten()

        dfs.append([cik, ticker, label, subset, fiscal_periods] + values.tolist())

    final_flatten_df = pd.DataFrame(dfs, columns=new_columns)
    final_flatten_df = final_flatten_df.reset_index(drop=True)
    return final_flatten_df


def get_train_val_test_split(X, y):
    X_train = X[X['subset'] == 'train']
    y_train = y[X['subset'] == 'train']

    X_val = X[X['subset'] == 'val']
    y_val = y[X['subset'] == 'val']

    X_test = X[X['subset'] == 'test']
    y_test = y[X['subset'] == 'test']

    X_train = X_train.drop('subset', axis=1)
    X_val = X_val.drop('subset', axis=1)
    X_test = X_test.drop('subset', axis=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def get_multimodal_data(drop_all_columns=False):
    # numerical dataset
    numerical_dataset = pd.read_csv('../dataset/numerical_dataset_version5_original.csv')
    numerical_dataset = flatten_financial_dataset(numerical_dataset)

    numerical_dataset_preprocessed = numerical_dataset.drop(['cik', 'ticker', 'Fiscal Period'], axis=1)

    X_numerical = numerical_dataset_preprocessed.drop('label', axis=1)
    y_numerical = numerical_dataset_preprocessed['label']

    X_numerical_train, y_numerical_train, X_numerical_val, y_numerical_val, X_numerical_test, y_numerical_test =(
        get_train_val_test_split(X_numerical, y_numerical))

    # textual dataset
    textual_dataset = pd.read_csv('../dataset/textual_data_version6_original.csv')
    textual_dataset_preprocessed = textual_dataset.drop(['cik'], axis=1)
    textual_dataset_preprocessed['text'] = textual_dataset_preprocessed['text'].apply(preprocess_text)

    X_textual = textual_dataset_preprocessed.drop('label', axis=1)
    y_textual = textual_dataset_preprocessed['label']

    if drop_all_columns:
        X_textual = X_textual.drop(['ticker', 'report_datetime'], axis=1)

    X_textual_train, y_textual_train, X_textual_val, y_textual_val, X_textual_test, y_textual_test = (
        get_train_val_test_split(X_textual, y_textual))

    return (X_numerical_train, y_numerical_train, X_numerical_val, y_numerical_val, X_numerical_test, y_numerical_test,
            X_textual_train, y_textual_train, X_textual_val, y_textual_val, X_textual_test, y_textual_test)

def get_train_data():
    (X_numerical_train, y_numerical_train, X_numerical_val, y_numerical_val, X_numerical_test, y_numerical_test,
     X_textual_train, y_textual_train, X_textual_val, y_textual_val, X_textual_test, y_textual_test) = get_multimodal_data()

    return X_numerical_train, y_numerical_train, X_textual_train, y_textual_train

def get_test_data():
    (X_numerical_train, y_numerical_train, X_numerical_val, y_numerical_val, X_numerical_test, y_numerical_test,
     X_textual_train, y_textual_train, X_textual_val, y_textual_val, X_textual_test, y_textual_test) = get_multimodal_data()

    tickers_test = X_textual_test['ticker']
    report_dates_test = X_textual_test['report_datetime']

    X_textual_test = X_textual_test.drop(['ticker', 'report_datetime'], axis=1)

    # preprocessing for finbert
    # y_textual_test = y_textual_test.values.astype(int)
    # y_textual_test = [1 if x == True else 0 for x in y_textual_test]

    return X_numerical_test, y_numerical_test, X_textual_test, y_textual_test, tickers_test, report_dates_test

def get_positive_samples_ticker_and_report_dates():
    textual_dataset = pd.read_csv('../dataset/textual_data_version6_original.csv')
    textual_dataset_preprocessed = textual_dataset.drop(['cik', 'text'], axis=1)

    X_textual = textual_dataset_preprocessed.drop('label', axis=1)
    y_textual = textual_dataset_preprocessed['label']

    positive_samples = X_textual[y_textual == 1]
    positive_samples = positive_samples.reset_index(drop=True)

    return positive_samples


def get_stock_data(ticker, report_date):
    report_date = pd.to_datetime(report_date)
    end_date = report_date + pd.DateOffset(days=10)
    start_date = report_date - pd.DateOffset(years=1)

    end_date = end_date.astype(str).values[0]
    start_date = start_date.astype(str).values[0]

    filename = f"../dataset/close-prices-for-test-data/{ticker.values[0]}_closeprice_gurufocus.csv"
    stock_close_prices = pd.read_csv(filename)
    stock_data = stock_close_prices[(stock_close_prices['date'] >= start_date) & (stock_close_prices['date'] <= end_date)]

    return stock_data