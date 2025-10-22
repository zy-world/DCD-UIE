# data/__init__.py

import torch.utils.data
import importlib
import logging

def create_dataloader(dataset, dataset_opt, phase):
    if phase in ['train', 'val']:
        batch_size = dataset_opt['batch_size'] if phase == 'train' else 1
        shuffle = dataset_opt['use_shuffle'] if phase == 'train' else False
        num_workers = dataset_opt['num_workers'] if phase == 'train' else 0

        try:
            from .__init__ import custom_collate_fn
            collate_fn = custom_collate_fn
            print("Using custom_collate_fn for dataloader.")
        except (ImportError, NameError):
            collate_fn = None
            print("custom_collate_fn not found or defined, using default dataloader behavior.")

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    mode = dataset_opt['mode']

    if mode == 'LRHR':
        from .LRHR_dataset import LRHRDataset as D
    else:

        raise NotImplementedError(f"Dataset mode [{mode}] is not supported.")

    dataset = D(
        dataroot_HR=dataset_opt['dataroot_HR'],
        dataroot_LR=dataset_opt['dataroot_LR'],
        datatype=dataset_opt['datatype'],
        l_resolution=dataset_opt['l_resolution'],
        r_resolution=dataset_opt['r_resolution'],
        split=phase,
        data_len=dataset_opt['data_len'],
        need_LR=dataset_opt.get('need_LR', False)
    )

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def custom_collate_fn(batch):

    hr_list = [item['HR'] for item in batch]
    sr_list = [item['SR'] for item in batch]
    text_feat_list = [item['text_feature'] for item in batch]
    index_list = [item['Index'] for item in batch]

    HR = torch.stack(hr_list)
    SR = torch.stack(sr_list)
    text_features = torch.stack(text_feat_list)
    Indices = torch.tensor(index_list, dtype=torch.long)
    
    return {'HR': HR, 'SR': SR, 'text_feature': text_features, 'Index': Indices}