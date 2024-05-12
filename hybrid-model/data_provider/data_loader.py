import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_BC_17_variables_5_years(Dataset):
    def __init__(self,
                 root_path,
                 flag='train',
                 numerical_data_path='bankrupt_companies_with_17_variables_5_years_version2_split.csv',
                 text_data_path='TODO',
                 scale=True,
                 batch_size=32,
                 company_observation_period=5):

        assert flag in ['train', 'test', 'val']
        self.set_type = flag

        self.scale = scale

        self.batch_size = batch_size
        self.company_observation_period = company_observation_period

        self.root_path = root_path
        self.numerical_data_path = numerical_data_path
        self.text_data_path = text_data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw_numerical = pd.read_csv(os.path.join(self.root_path, self.numerical_data_path))
        df_numerical = df_raw_numerical[df_raw_numerical['subset'] == self.set_type]

        df_raw_textual = pd.read_csv(os.path.join(self.root_path, self.text_data_path))
        df_textual = df_raw_textual[df_raw_textual['subset'] == self.set_type]

        # TODO: match text with numerical data by cik

        df_data_x = df_numerical.drop(columns=['cik', 'ticker', 'label', 'subset', 'Fiscal Period']).to_numpy()

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data_x)
            df_data_x = scaler.transform(df_data_x)

        data_x = self.__get_data_grouped_by_years__(df_data_x)

        data_y = df_numerical.groupby('cik').agg({'label': 'first'}).reset_index()
        data_y = data_y['label'].astype(int).to_numpy()

        self.data_x = data_x
        self.data_y = data_y

        # custom batching mechanism - each object has `company_observation_period` number of observations,
        # so we cannot use DataLoader
        self.batched_x, self.batched_y = self.__create_batches_from_data__(batch_size=self.batch_size)

    def __get_data_grouped_by_years__(self, array):

        variables_count = array.shape[1]

        grouped_by_years_array = array.reshape(-1, self.company_observation_period, variables_count)

        return grouped_by_years_array

    def __create_batches_from_data__(self, batch_size):
        # create batched X from data
        x = self.data_x

        company_observation_period = x.shape[1]
        variables_count = x.shape[2]

        complete_batches_count = x.shape[0] // batch_size

        X_aligned_to_batch_size = x[:complete_batches_count * batch_size]
        batched_X = X_aligned_to_batch_size.reshape(-1, batch_size, company_observation_period, variables_count)

        # create batched y from data
        y = self.data_y

        y_aligned_to_batch_size = y[:complete_batches_count * batch_size]
        batched_y = y_aligned_to_batch_size.reshape(-1, batch_size, 1)

        return batched_X, batched_y

    def __getitem__(self, index):
        return self.batched_x[index], self.batched_y[index], # TODO: add text data

    def __len__(self):
        return len(self.batched_x)
