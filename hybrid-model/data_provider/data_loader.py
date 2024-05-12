import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
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


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)