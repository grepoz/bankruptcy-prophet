from data_provider.data_loader import Dataset_BC_17_variables_5_years
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_BC_17_variables_5_years

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        numerical_data_path=args.numerical_data_path,
        raw_textual_data_path=args.raw_textual_data_path,
        batch_size=args.batch_size,
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    return data_set, data_loader
