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