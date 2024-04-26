from torch.utils.data import DataLoader
from .data_loaders import Dataset_Electricity


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.data.batch_size

    dataset = Dataset_Electricity(args, flag)
    print(flag, len(dataset))

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.data.num_workers,
        drop_last=drop_last
    )

    return dataset, data_loader