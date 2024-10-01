from data_provider.data_loader import Dataset_BC_17_variables_5_years
from torch.utils.data import DataLoader


def data_provider(args, flag, scaler):
    Data = Dataset_BC_17_variables_5_years

    if flag == 'train':
        batch_size = args.batch_size
    else:
        batch_size = args.test_batch_size

    data_set = Data(
        root_path=args.root_path,
        numerical_data_path=args.numerical_data_path,
        raw_textual_data_path=args.raw_textual_data_path,
        flag=flag,
        batch_size=batch_size,
        scaler=scaler
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=True)

    return data_set, data_loader
