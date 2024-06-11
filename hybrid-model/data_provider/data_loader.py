import os
import numpy as np
import pandas as pd
import string
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import random
from data_provider.encode_text import encode_roberta

warnings.filterwarnings('ignore')


class Dataset_BC_17_variables_5_years(Dataset):
    def __init__(self,
                 root_path,
                 flag='train',
                 numerical_data_path='bankrupt_companies_with_17_variables_5_years_version2_split_matched_with_reports.csv',
                 raw_textual_data_path='textual_data_matched_with_fin_data_preprocessed.csv',
                 scale=True,
                 batch_size=16,
                 company_observation_period=5,
                 use_cached_textual_data=True,
                 textual_data_encoding_size=768):
        assert flag in ['train', 'test', 'val']
        self.set_type = flag

        self.scale = scale

        self.batch_size = batch_size
        self.company_observation_period = company_observation_period
        self.use_cached_textual_data = use_cached_textual_data
        self.textual_data_encoding_size = textual_data_encoding_size

        self.root_path = root_path
        self.numerical_data_path = numerical_data_path
        self.raw_textual_data_path = raw_textual_data_path
        self.encoded_textual_data_dirpath = './data/bankrupt_companies_with_17_variables_5_years/textual_data/encoded_corpora/'
        self.encoded_textual_data_filepath = 'textual_data_encoded_ulti_representations_cls'

        self.__read_data__()

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.digits))
        return text

    def __read_data__(self):
        df_raw_numerical = pd.read_csv(os.path.join(self.root_path, self.numerical_data_path))
        df_numerical = df_raw_numerical[df_raw_numerical['subset'] == self.set_type]

        # df_shuffled = self.shuffle_data_by_object(df_numerical)

        df_raw_textual = pd.read_csv(os.path.join(self.root_path, self.raw_textual_data_path))
        df_textual = df_raw_textual[df_raw_textual['subset'] == self.set_type]
        df_textual['text'] = df_textual['text'].apply(self.preprocess_text)

        df_numerical = df_numerical[df_numerical['cik'].isin(df_textual['cik'])]
        df_numerical_unique_cik = df_numerical['cik'].unique()
        assert len(df_numerical_unique_cik) == len(df_textual)

        # data are already sorted by cik and Fiscal Period. Furthermore text data is already aligned by cik with numerical data

        df_data_x = df_numerical.drop(columns=['cik', 'ticker', 'label', 'subset', 'Fiscal Period']).to_numpy()

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data_x)
            df_data_x = scaler.transform(df_data_x)

        self.data_x = self.__get_data_grouped_by_years__(df_data_x)
        self.data_y = df_numerical[df_numerical.index % 5 == 0].reset_index(drop=True)['label'].astype(int).to_numpy()

        textual_data_loaded = False
        ulti_representations_cls = None
        if self.use_cached_textual_data:
            try:
                ulti_representations_cls = np.load(f'{self.encoded_textual_data_dirpath}{self.encoded_textual_data_filepath}_{self.set_type}.npy')
                textual_data_loaded = True
            except FileNotFoundError:
                print('Cached textual data not found, encoding textual data from scratch')
                textual_data_loaded = False

        if not textual_data_loaded:
            ulti_representations, ulti_representations_cls = self.__encode_textual_data__(df_textual['text'])

            np.save(f'{self.encoded_textual_data_dirpath}textual_data_encoded_ulti_representations_{self.set_type}.npy', ulti_representations)
            np.save(f'{self.encoded_textual_data_dirpath}{self.encoded_textual_data_filepath}_{self.set_type}.npy', ulti_representations_cls)

        max_len = 1
        pos = np.zeros((len(ulti_representations_cls), max_len, 768))

        ulti_representations_cls = ulti_representations_cls.reshape(-1, 1, 768)

        for idx, doc in enumerate(ulti_representations_cls):
            if doc.shape[0] <= max_len:
                pos[idx][:doc.shape[0], :] = doc
            else:
                pos[idx][:max_len, :] = doc[:max_len, :]

        self.data_x_textual = pos

        # 1000, 20, 768 | 1000

        assert len(self.data_x_textual[0][0]) == self.textual_data_encoding_size

    def __get_data_grouped_by_years__(self, array):
        variables_count = array.shape[1]

        grouped_by_years_array = array.reshape(-1, self.company_observation_period, variables_count)

        return grouped_by_years_array

    def __create_batches_from_data__(self, batch_size):
        def _get_reshaped_data(data):
            return data[:complete_batches_size].reshape(-1, batch_size, 1)

        # create batched X from data
        x = self.data_x

        company_observation_period = x.shape[1]
        variables_count = x.shape[2]

        complete_batches_count = x.shape[0] // batch_size
        complete_batches_size = complete_batches_count * batch_size

        X_aligned_to_batch_size = x[:complete_batches_size]
        batched_X = X_aligned_to_batch_size.reshape(-1, batch_size, company_observation_period, variables_count)

        y = self.data_y
        batched_y = _get_reshaped_data(y)

        batched_x_textual = self.data_x_textual[:complete_batches_size].reshape(-1, batch_size, 1, self.textual_data_encoding_size)

        return batched_X, batched_y, batched_x_textual

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.data_x_textual[index]

    def __len__(self):
        return len(self.data_y)

    def __encode_textual_data__(self, data):
        # use roberta to encode textual data into high-dimensional vectorised representations
        # https://github.com/GeorgeLuImmortal/Hierarchical-BERT-Model-with-Limited-Labelled-Data#:~:text=Step%201.%20Data%20Processing

        encoded_textual_data = encode_roberta(data)

        return encoded_textual_data

    def shuffle_data_by_object(self, df):

        data = df.values.tolist()
        shuffled_data = []
        object_data = {}

        for row in data:
            cik = row[0]
            if cik not in object_data:
                object_data[cik] = []
            object_data[cik].append(row)

        shuffled_object_keys = list(object_data.keys())
        random.shuffle(shuffled_object_keys)

        for cik in shuffled_object_keys:
            shuffled_data.extend(object_data[cik])

        shuffled_df = pd.DataFrame(shuffled_data, columns=df.columns).reset_index(drop=True)
        return shuffled_df