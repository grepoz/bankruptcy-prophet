import pandas as pd

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

def get_test_data():
    financial_dataset = pd.read_csv('bankrupt_companies_with_17variables_5years_split_version5_complete.csv')
    financial_dataset = flatten_financial_dataset(financial_dataset)

    financial_dataset_preprocessed = financial_dataset.drop(['cik', 'Fiscal Period'], axis=1)

    X = financial_dataset_preprocessed.drop('label', axis=1)
    y = financial_dataset_preprocessed['label']

    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_split(X, y)

    tickers_test = X_test['ticker']

    X_test = X_test.drop('ticker', axis=1)

    return X_test, y_test, tickers_test