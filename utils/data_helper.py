import os
import glob
import pickle

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .labels import CLASSES, R_CLASSES


class MIMICCXR(Dataset):
    def __init__(self, cxr_data_dir, cxr_fnames, split='train', cxr_pkl_fpath=None):
        self.data_dir = cxr_data_dir
        self.filename_list = cxr_fnames

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if cxr_pkl_fpath is not None and os.path.isfile(cxr_pkl_fpath):
            print(f'Loading CXR file from {cxr_pkl_fpath}')
            with open(cxr_pkl_fpath, 'rb') as f:
                self.img_preloaded = pickle.load(f)
        else:
            print(f'Pre-stored pkl file is not found, loading raw CXR data...')
            paths = glob.glob(f'{cxr_data_dir}/**/*.jpg', recursive=True)
            filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}
            self.img_preloaded = {}
            for fn in self.filename_list:
                img = Image.open(filenames_to_path[fn]).convert('RGB')
                self.img_preloaded[fn] = img
            if cxr_pkl_fpath is not None:
                with open(cxr_pkl_fpath, 'wb') as f:
                    pickle.dump(self.img_preloaded, f)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.filename_list[index]
        img = self.img_preloaded[index]
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.filename_list)


class EHRdataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile,
                 dataset_dir, period_length=48.0,
                 ehr_pkl_fpath=None):
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length

        self._dataset_dir = dataset_dir
        #listfile: the csv contains the data/label for the specific task
        with open(listfile, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]


        self._data = [line.split(',') for line in self._data]
        #mas[0]: dir, mas[1]: time, mas[2]: stay id, mas[3:]: y for specific task
        self.data_map = {
            mas[0]: {
                'labels': list(map(float, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }
        self.names = list(self.data_map.keys())
        if ehr_pkl_fpath is not None and os.path.isfile(ehr_pkl_fpath):
            print(f'Loading EHR data from {ehr_pkl_fpath}')
            with open(ehr_pkl_fpath, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print(f'Pre-stored pkl file is not found, loading raw EHR data...')
            self.processed_data = {}
            for k in self.names:
                ret = self.read_by_file_name(k)
                data = ret["X"]
                ts = ret["t"] if ret['t'] > 0.0 else self._period_length
                ys = ret["y"]
                names = ret["name"]
                data = self.discretizer.transform(data, end=ts)[0]
                if (self.normalizer is not None):
                    data = self.normalizer.transform(data)
                ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
                self.processed_data[k] = {'data': data, 'ys': ys}

            if ehr_pkl_fpath is not None:
                with open(ehr_pkl_fpath, 'wb') as f:
                    pickle.dump(self.processed_data, f)

    def _read_timeseries(self, ts_filename, time_bound=None):

        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_by_file_name(self, index, time_bound=None):
        # t = self.data_map[index]['time'] if time_bound is None else time_bound
        t = min(self.data_map[index]['time'], 48)
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        # (X, header) = self._read_timeseries(index, time_bound=time_bound)
        (X, header) = self._read_timeseries(index, time_bound=48)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}


    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.processed_data[index]
        data, ys = ret['data'], ret['ys']
        data[data > 10] = 0  # some missing data represented as 9999999, causing NaN; Filter them out
        data[data < -10] = 0
        return data, ys


    def __len__(self):
        return len(self.processed_data)


def get_ehr_datasets(discretizer, normalizer, ehr_data_dir,
                     ehr_pkl_fpath_train=None, ehr_pkl_fpath_val=None, ehr_pkl_fpath_test=None,
                     list_file_name='ehr_list_phenotyping_48h'):
    train_ds = EHRdataset(discretizer, normalizer,
                          listfile=ehr_data_dir/(list_file_name+'_train.csv'),
                          dataset_dir=ehr_data_dir/'train',
                          ehr_pkl_fpath=ehr_pkl_fpath_train)
    val_ds = EHRdataset(discretizer, normalizer,
                        listfile=ehr_data_dir/(list_file_name+'_val.csv'),
                        dataset_dir=ehr_data_dir/'train',
                        ehr_pkl_fpath=ehr_pkl_fpath_val)
    test_ds = EHRdataset(discretizer, normalizer,
                         listfile=ehr_data_dir/(list_file_name+'_test.csv'),
                         dataset_dir=ehr_data_dir/'test',
                         ehr_pkl_fpath=ehr_pkl_fpath_test)
    return train_ds, val_ds, test_ds

# def my_collate_ehr(batch):
#     x = [item[0] for item in batch]
#     x, seq_length = pad_zeros(x)
#     targets = np.array([item[1] for item in batch])
#     return [x, targets, seq_length]


def load_discretized_header(discretizer, ehr_dir, txt_fpath=None):
    if txt_fpath is not None and os.path.isfile(txt_fpath):
        with open(txt_fpath, 'r') as f:
            with open(txt_fpath, 'r') as f:
                header = [x.strip() for x in f.readlines()]
    else:
        path = ehr_dir / 'phenotyping_raw_ts/train/14991576_episode3_timeseries.csv'
        ret = []
        with open(path, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
            ret = np.stack(ret)
        header = discretizer.transform(ret)[1].split(',')

        if txt_fpath is not None:
            with open(txt_fpath, 'w') as f:
                f.write('\n'.join(header))
    return header



class MIMIC_CXR_EHR(Dataset):
    def __init__(self, data_pairs, data_ratio, metadata_with_labels,
                 ehr_ds, cxr_ds, split='train'):

        self.CLASSES = CLASSES
        # if 'radiology' in args.labels_set:
        #     self.CLASSES = R_CLASSES

        self.metadata_with_labels = metadata_with_labels
        self.cxr_files_paired = self.metadata_with_labels.dicom_id.values
        self.ehr_files_paired = (self.metadata_with_labels['stay'].values)
        # self.cxr_files_all = cxr_ds.filenames_loaded
        self.ehr_files_all = ehr_ds.names
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds
        self.data_pairs = data_pairs
        self.split = split
        self.data_ratio = data_ratio

    def __getitem__(self, index):
        if self.data_pairs == 'paired':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data = self.cxr_ds[self.cxr_files_paired[index]]
            return ehr_data, cxr_data, labels_ehr

        else:
            if index < len(self.ehr_files_paired):
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
                cxr_data = self.cxr_ds[self.cxr_files_paired[index]]
            else:
                # index = random.randint(0, len(self.ehr_files_unpaired)-1)
                index = self.ehr_files_unpaired[index-len(self.ehr_files_paired)]
                ehr_data, labels_ehr = self.ehr_ds[index]
                cxr_data = None
            return ehr_data, cxr_data, labels_ehr

    def __len__(self):
        if self.data_pairs == 'paired':
            return len(self.ehr_files_paired)
        else:
            return len(self.ehr_files_paired) + int(self.data_ratio * len(self.ehr_files_unpaired))



def load_cxr_ehr(cxr_resized_data_dir, data_pairs_train,
                 meta_pkl_fpath, ehr_datasets, data_ratio, batch_size,
                 cxr_pkl_fpath_train, cxr_pkl_fpath_val, cxr_pkl_fpath_test,
                 num_workers, data_pairs_val='partial'):

    ehr_train_ds, ehr_val_ds, ehr_test_ds = ehr_datasets
    with open(meta_pkl_fpath, 'rb') as f:
        metas_with_labels = pickle.load(f)

    train_meta_with_labels = metas_with_labels['train']
    val_meta_with_labels = metas_with_labels['val']
    test_meta_with_labels = metas_with_labels['test']

    cxr_train_ds = MIMICCXR(cxr_data_dir=cxr_resized_data_dir,
                            cxr_fnames=train_meta_with_labels.dicom_id.tolist(),
                            split='train',
                            cxr_pkl_fpath=cxr_pkl_fpath_train)
    cxr_val_ds = MIMICCXR(cxr_data_dir=cxr_resized_data_dir,
                          cxr_fnames=val_meta_with_labels.dicom_id.tolist(),
                          split='val',
                          cxr_pkl_fpath=cxr_pkl_fpath_val)
    cxr_test_ds = MIMICCXR(cxr_data_dir=cxr_resized_data_dir,
                           cxr_fnames=test_meta_with_labels.dicom_id.tolist(),
                           split='test',
                           cxr_pkl_fpath=cxr_pkl_fpath_test)

    # train data loader
    train_ds = MIMIC_CXR_EHR(data_pairs=data_pairs_train, data_ratio=data_ratio,
                             metadata_with_labels=train_meta_with_labels,
                             ehr_ds=ehr_train_ds, cxr_ds=cxr_train_ds, split='train')
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          collate_fn=my_collate_cxr_ehr, pin_memory=True,
                          num_workers=num_workers, drop_last=True)

    # validation data loader
    val_ds = MIMIC_CXR_EHR(data_pairs=data_pairs_val, data_ratio=data_ratio,
                           metadata_with_labels=val_meta_with_labels,
                           ehr_ds=ehr_val_ds, cxr_ds=cxr_val_ds, split='val')
    val_dl = DataLoader(val_ds, batch_size, shuffle=False,
                        collate_fn=my_collate_cxr_ehr, pin_memory=True,
                        num_workers=num_workers, drop_last=False)

    # partial test datasets
    test_ds_partial = MIMIC_CXR_EHR(data_pairs='partial', data_ratio=data_ratio,
                                    metadata_with_labels=test_meta_with_labels,
                                    ehr_ds=ehr_test_ds, cxr_ds=cxr_test_ds, split='test')
    test_dl_partial = DataLoader(test_ds_partial, batch_size, shuffle=False,
                                 collate_fn=my_collate_cxr_ehr, pin_memory=True,
                                 num_workers=num_workers, drop_last=False)

    test_ds_paired = MIMIC_CXR_EHR(data_pairs='paired', data_ratio=data_ratio,
                                   metadata_with_labels=test_meta_with_labels,
                                   ehr_ds=ehr_test_ds, cxr_ds=cxr_test_ds, split='test')
    test_dl_paired = DataLoader(test_ds_paired, batch_size, shuffle=False,
                                collate_fn=my_collate_cxr_ehr, pin_memory=True,
                                num_workers=num_workers, drop_last=False)


    return train_dl, val_dl, test_dl_partial, test_dl_paired

# def printPrevalence(merged_file, args):
#     if args.labels_set == 'pheno':
#         total_rows = len(merged_file)
#         print(merged_file[CLASSES].sum()/total_rows)
#     else:
#         total_rows = len(merged_file)
#         print(merged_file['y_true'].value_counts())
    # import pdb; pdb.set_trace()


def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    #time sequence length: different for each patient.
    seq_length = [x.shape[0] for x in arr]
    #x.shape[1:] are length of variables
    max_len = max(seq_length)
    #wow magic! (558,) + (76,) tuple's + is concatenation. --> (558, 76)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
        #if not enough, make it up to min_length
        #TODO: if any change to upper part, remember to change this part too!!!!!!!
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length


def my_collate_cxr_ehr(batch):
    #x: all ehr data
    x = [item[0] for item in batch]
    #pairs: False if no cxr. True if cxr exists
    pairs = [False if item[1] is None else True for item in batch]
    # if cxr missing: use 0.
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch]) #ehr labels
    # targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch]) #cxr labels
    return [x, img, targets_ehr, seq_length, pairs]
