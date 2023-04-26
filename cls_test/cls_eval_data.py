import os
import glob
import h5py
import numpy as np
import random
from torch.utils.data import Dataset


def load_modelnet_data(partition):
    BASE_DIR = r'/home/ies/qhuang/H_AttentionProject/cls_val_dataset'
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet40')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_modelnet10_data(partition):
    BASE_DIR = r'/home/ies/qhuang/H_AttentionProject/cls_val_dataset'
    DATA_DIR = os.path.join(BASE_DIR, 'modelnet10')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet10_hdf5_2048', '%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_ScanObjectNN(partition):
    BASE_DIR = r'/home/ies/qhuang/H_AttentionProject/cls_val_dataset/scanobjectnn/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label


# get index list
def get_index_by_percent(data, percent):
    total = data.shape[0]
    percent_num = int(total * percent / 100)
    original_index_list = [i for i in range(total)]
    random.shuffle(original_index_list)
    index_list = original_index_list[:percent_num]
    return index_list


def load_modelnet_data_percent(partition, percent=1):
    all_data, all_label = load_modelnet_data(partition)
    print("We are using {} percent data to {}.".format(percent, partition))
    if percent == 100 or partition == 'test':
        return all_data, all_label
    else:
        index_list = get_index_by_percent(all_label, percent)
        return all_data[index_list], all_label[index_list]


def load_modelnet10_data_percent(partition, percent=1):
    all_data, all_label = load_modelnet10_data(partition)
    print("We are using {} percent data to {}.".format(percent, partition))
    if percent == 100 or partition == 'test':
        return all_data, all_label
    else:
        index_list = get_index_by_percent(all_label, percent)
        return all_data[index_list], all_label[index_list]


def load_ScanObjectNN_percent(partition, percent=1):
    all_data, all_label = load_ScanObjectNN(partition)
    print("We are using {} percent data to {}.".format(percent, partition))
    if percent == 100 or partition == 'test':
        return all_data, all_label
    else:
        index_list = get_index_by_percent(all_label, percent)
        return all_data[index_list], all_label[index_list]


# For normal ModelNet, ScanObjectMM
class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet10SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet10_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


# For dataset select with percentage
class ModelNet40(Dataset):
    def __init__(self, num_points=1024, partition='train', percent=100):
        self.data, self.label = load_modelnet_data_percent(partition, percent)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet10(Dataset):
    def __init__(self, num_points=1024, partition='train', percent=100):
        self.data, self.label = load_modelnet10_data_percent(partition, percent)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN(Dataset):
    def __init__(self, num_points=1024, partition='train', percent=100):
        self.data, self.label = load_ScanObjectNN_percent(partition, percent)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    def data_test_1():
        train_data, train_label = load_modelnet_data('train')
        test_data, test_label = load_modelnet_data('test')
        print('Train data size: ', train_data.shape)  # 9840, 2048, 3
        print('Test data size: ', test_data.shape)  # 2468, 2048, 3
        for i in range(train_label.shape[0]):
            print(train_label[i])


    def data_test_2():
        train_data, train_label = load_ScanObjectNN('train')
        test_data, test_label = load_ScanObjectNN('test')
        print('Train data size: ', train_data.shape)  # 2309, 2048, 3
        print('Test data size: ', test_data.shape)  # 581, 2048, 3


    def data_test_3():
        train_data, train_label = load_modelnet10_data('train')
        test_data, test_label = load_modelnet10_data('test')
        print('Train data size: ', train_data.shape)  # 3991, 2048, 3
        print('Test data size: ', test_data.shape)  # 908, 2048, 3


    def data_test_4(p):
        train_data, train_label = load_ScanObjectNN_percent('train', p)
        test_data, test_label = load_ScanObjectNN_percent('test', p)
        print('Train data size: ', train_data.shape)  # 3991, 2048, 3
        print('Test data size: ', test_data.shape)  # 908, 2048, 3


    data_test_4(20)
    # data_test_4(20)
