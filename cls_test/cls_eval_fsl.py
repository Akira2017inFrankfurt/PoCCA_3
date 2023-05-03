import os
import torch
from network.basic_encoders import PointNet_CLS_Encoder, DGCNN_CLS_Encoder
from cls_eval_data import load_modelnet_data, load_ScanObjectNN
import argparse
from tqdm import tqdm
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from cls_model_4_compare import get_encoder


def get_local_encoder(args):
    if args.model_choice == 0:
        encoder = PointNet_CLS_Encoder().to(device)
    else:
        encoder = DGCNN_CLS_Encoder().to(device)
    local_encoder = get_encoder(encoder, args.weight_path)
    return local_encoder.to(device)


def load_data(args):
    data_func = load_modelnet_data
    n_cls = 40
    if args.dataset is 'scan':
        data_func = load_ScanObjectNN
        n_cls = 15
    train_point, train_label = data_func('train')
    test_point, test_label = data_func('test')

    # prepare for few shot learning
    label_idx = {}
    for key in range(n_cls):
        label_idx[key] = []
        for i, label in enumerate(train_label):
            if label == key:
                label_idx[key].append(i)

    return train_point, train_label, test_point, test_label, label_idx, n_cls


def print_info(args):
    _, local_model_name = os.path.split(args.weight_path)
    print('- Model: ', local_model_name[:-4])
    print('- Dataset: ', args.dataset)
    print('- {} way {} shot\n'.format(args.k_way, args.m_shot))


def train_svm(args):
    print_info(args)
    data_train, label_train, data_test, label_test, label_idx, n_cls = load_data(args)
    acc = []
    for run in tqdm(range(args.n_runs)):
        k = args.k_way
        m = args.m_shot
        n_q = args.n_query
        k_way = random.sample(range(n_cls), k)

        data_support = []
        label_support = []
        data_query = []
        label_query = []

        for i, class_id in enumerate(k_way):
            support_id = random.sample(label_idx[class_id], m)
            query_id = random.sample(list(set(label_idx[class_id]) - set(support_id)), n_q)

            pc_support_id = data_train[support_id]
            pc_query_id = data_train[query_id]
            data_support.append(pc_support_id)
            label_support.append(i * np.ones(m))
            data_query.append(pc_query_id)
            label_query.append(i * np.ones(n_q))

        data_support = np.concatenate(data_support)
        label_support = np.concatenate(label_support)
        data_query = np.concatenate(data_query)
        label_query = np.concatenate(label_query)

        feats_train = []
        labels_train = []
        model = get_local_encoder(args)
        model = model.eval()

        for i in range(k * m):
            data = torch.from_numpy(np.expand_dims(data_support[i], axis=0))
            label = int(label_support[i])
            data = data.to(device)
            # data = torch.cat((data, data))
            with torch.no_grad():
                feat = model(data)
            feat = feat.detach().cpu().numpy().tolist()

            feats_train.append(feat)
            labels_train.append(label)

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i in range(k * n_q):
            data = torch.from_numpy(np.expand_dims(data_query[i], axis=0))
            label = int(label_query[i])
            data = data.to(device)
            # data = torch.cat((data, data))
            with torch.no_grad():
                feat = model(data)
            feat = feat.detach().cpu().numpy().tolist()
            feats_test.append(feat)
            labels_test.append(label)

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        feats_train, feats_test = feats_train.squeeze(), feats_test.squeeze()
        # print('feats train shape: ', feats_train.shape)
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feats_train)
        model_tl = SVC(kernel='linear')
        model_tl.fit(scaled, labels_train)
        test_scaled = scaler.transform(feats_test)

        accuracy = model_tl.score(test_scaled, labels_test) * 100
        acc.append(accuracy)

    print(f'{np.mean(acc)} +/- {np.std(acc)}')
    print('************************************')


if __name__ == "__main__":
    dataset_list = ['modelnet40', 'scan']
    n_way_list = [5, 10]
    k_shot_list = [10, 20]
    for dataset in dataset_list:
        for n_way in n_way_list:
            for k_shot in k_shot_list:
                def parameters_init():
                    parser = argparse.ArgumentParser(description='Point Cloud Classification Evaluation!')
                    parser.add_argument('--model_choice', type=int, default=1, help='0 for PN, 1 for DGCNN')
                    parser.add_argument('--use_pretrain', type=bool, default=True)
                    parser.add_argument('--weight_path', type=str, default=r'/home/haruki/下载/SimAttention/scripts/weights/cls_dg_1205_cross_99.pth')
                    # parser.add_argument('--weight_path', type=str, default=r'/home/haruki/下载/SimAttention/scripts/weights/cls_dg_1125_multi_scale_99.pth')
                    parser.add_argument('--freeze_encoder', type=bool, default=True)
                    parser.add_argument('--dataset', type=str, default=dataset, choices=['modelnet40', 'scan'])
                    parser.add_argument('--n_runs', type=int, default=10)
                    parser.add_argument('--k_way', type=int, default=n_way)
                    parser.add_argument('--m_shot', type=int, default=k_shot)
                    parser.add_argument('--n_query', type=int, default=20)

                    parsers = parser.parse_args()
                    return parsers
                hyper_parameters = parameters_init()

                device = torch.device('cuda')
                train_svm(args=hyper_parameters)
