import argparse
import os
import time

import torch

from utils import fact, parse_str_dim, get_class_list
import numpy as np
from datasets import TrafficDataset
from models import FingerprintLSTM


def data_subset(data_dir, class_choice, y_test, y_pred, feature_importance, subset=30):
    """
    subset the data so accelerate the metric speed
    :param data_dir: str
    :param class_choice: str all or list of selected class names
    :param y_test: ndarray
    :param y_pred: ndarray
    :param feature_importance: n_samples x feature_importance
    :param subset: int ratio (test_ratio is 0.3 so 30 will use 0.3/30 data in the total data per class)
    :return:
    """
    class_list = get_class_list(data_dir, class_choice)

    # subset
    y_pred_subset_list = []  # don't know the final num so save to the list first
    feature_importance_subset_list = []
    total = 0
    for i in range(len(class_list)):
        c_num = y_test[y_test == i].shape[0]
        c_select_num = c_num // subset
        total += c_select_num
        c_y_pred = y_pred[y_test == i][:c_select_num]
        c_feature_importance = feature_importance[y_test == i][:c_select_num]
        y_pred_subset_list.append(c_y_pred)
        feature_importance_subset_list.append(c_feature_importance)

    # merge
    y_pred_subset = np.zeros(total)
    feature_importance_subset = np.zeros((total, feature_importance.shape[1]))
    p = 0
    for i in range(len(y_pred_subset_list)):
        tmp = y_pred_subset_list[i]
        y_pred_subset[p:p+tmp.shape[0]] = tmp[:]
        tmp = feature_importance_subset_list[i]
        feature_importance_subset[p:p+tmp.shape[0]] = \
            tmp[:, :]
        p += tmp.shape[0]

    return y_pred_subset, feature_importance_subset


def interpret_similarity(feature_importance_a, feature_importance_b, k):
    """
    :param feature_importance_a: ndarray (feature_num,)
    :param feature_importance_b: ndarray (feature_num,)
    :param k: int
    :return:
    """
    fea_a_with_index = []
    fea_b_with_index = []
    for i in range(feature_importance_a.shape[0]):
        fea_a_with_index.append((i, abs(feature_importance_a[i])))
        fea_b_with_index.append((i, abs(feature_importance_b[i])))

    fea_a_with_index = sorted(fea_a_with_index, key=lambda e: e[1], reverse=True)
    fea_b_with_index = sorted(fea_b_with_index, key=lambda e: e[1], reverse=True)

    # calculation intersection
    fea_a_top_k = fea_a_with_index[:k]
    fea_a_bucket = np.zeros(feature_importance_a.shape[0])
    for e in fea_a_top_k:
        fea_a_bucket[e[0]] = 1
    fea_b_top_k = fea_b_with_index[:k]
    fea_b_bucket = np.zeros(feature_importance_b.shape[0])
    for e in fea_b_top_k:
        fea_b_bucket[e[0]] = 1

    # only place with 1 should be calculated
    a_and_b = ((fea_a_bucket == fea_b_bucket) * fea_a_bucket * fea_b_bucket).sum()

    return 2 * a_and_b / (k + k)


def interpret_stability(similar_model_importance_list, k):
    """
    :param similar_model_importance_list: list of ndarray, each is n_samples x feature_num
    :param k: int
    :return:
    """
    stability = 0

    for i in range(len(similar_model_importance_list)):
        for j in range(len(similar_model_importance_list)):
            if i > j:
                a, b = similar_model_importance_list[i], similar_model_importance_list[j]
                for p in range(a.shape[0]):
                    start = time.time()
                    stability += interpret_similarity(a[p, :], b[p, :], k)
                    end = time.time()
                    print('[DONE] {0}/{1} Time cost (s): {2:.4f}'.format(p+1, a.shape[0], end-start))

    stability /= (fact(len(similar_model_importance_list)) / 2)  # C(a, 2)
    stability /= similar_model_importance_list[0].shape[0]  # sample num

    print('Stability: {0:.4f}'.format(stability))
    return stability


def interpret_robustness(feature_importance, y_pred, k):
    """
    :param feature_importance: ndarray, n_samples x n_features
    :param y_pred: ndarray
    :param k: int
    :return:
    """
    robustness = 0

    for i in range(y_pred.shape[0]):
        start = time.time()

        ei = feature_importance[i, :]
        sxi = feature_importance[y_pred == y_pred[i]]
        dxi = feature_importance[y_pred != y_pred[i]]

        avg_sxi = 0
        for j in range(sxi.shape[0]):
            avg_sxi += interpret_similarity(ei, sxi[j], k)
        avg_sxi /= sxi.shape[0]

        avg_dxi = 0
        for j in range(dxi.shape[0]):
            avg_dxi += interpret_similarity(ei, dxi[j], k)
        avg_dxi /= dxi.shape[0]

        robustness += avg_sxi - avg_dxi

        end = time.time()
        print('[DONE] {0}/{1} Time cost (s): {2:.4f}'.format(i + 1, y_pred.shape[0], end - start))

    robustness /= y_pred.shape[0]
    print('Robustness: {0:.4f}'.format(robustness))

    return y_pred


def interpret_effectiveness(feature_importance, k, y_pred, dataset, model):
    """
    :param feature_importance: ndarray, n_samples x n_features
    :param k: int
    :param y_pred: ndarray
    :param dataset: torch dataset
    :param model: torch model implement forward
    :return:
    """
    effectiveness = 0
    for i in range(feature_importance.shape[0]):
        start = time.time()

        # get top k features
        fea_with_index = []
        for j in range(feature_importance.shape[1]):
            fea_with_index.append((j, feature_importance[i, j]))
        fea_with_index = sorted(fea_with_index, key=lambda e: e[1], reverse=True)
        top_k_index = []
        for j in range(k):
            top_k_index.append(fea_with_index[j][0])
        # mutate top k features
        x, y = dataset.data[i], dataset.label[i]
        x = x.unsqueeze(0)
        x[:, :, top_k_index] = 0
        out = model(x)
        pred = torch.argmax(out, dim=1)
        if pred != y_pred[i]:
            effectiveness += 1

        end = time.time()
        print('[DONE] {0}/{1} Time cost (s): {2:.4f}'.format(i + 1, feature_importance.shape[0], end - start))

    effectiveness /= y_pred.shape[0]
    print('Effectiveness: {0:.4f}'.format(effectiveness))

    return effectiveness


def cal_stability(similar_model_importance_files, k):
    """
    :param similar_model_importance_files: str of all files connected by ',', 'a.npy,b.npy,c.npy'
    :param k: int
    :return:
    """
    file_list = similar_model_importance_files.split(',')
    similar_model_importance_list = []
    for f in file_list:
        similar_model_importance_list.append(np.load(f))
    interpret_stability(similar_model_importance_list, k)


def cal_robustness(feature_importance, y_pred, k, data_dir, class_choice, y_test, subset=30):
    """
    :param feature_importance: str full path of feature importance file
    :param y_pred: str full path of y_pred file
    :param k: int
    :param data_dir: str
    :param class_choice: str all or str list of class names
    :param y_test: str full path of y_test file
    :param subset: bool
    :return:
    """
    feature_importance = np.load(feature_importance)
    y_pred = np.load(y_pred)
    y_test = np.load(y_test)
    if subset != 1:
        y_pred, feature_importance = data_subset(data_dir, class_choice, y_test, y_pred, feature_importance, subset)
    interpret_robustness(feature_importance, y_pred, k)


def cal_effectiveness(feature_importance, k, y_pred, data_dir, class_choice, scale, fields_file,
                      input_dims, encoding_dims, seq_dims, model_path, model_name, gpu=False,
                      feature_choice=None, encoding=False):
    """
    :param feature_importance: str full path of feature importance file
    :param k: int
    :param y_pred: str full path of y_pred file
    :param data_dir: str
    :param class_choice: all or str list of chosen class names
    :param scale: str of scale file without suffix
    :param fields_file: str full path of fields file
    :param input_dims: int or list of int
    :param encoding_dims: int or list of int
    :param seq_dims: int or list of int
    :param model_path: str
    :param model_name: str
    :param gpu: bool
    :param feature_choice: str prefix of feature choice without suffix.joblib
    :param encoding: bool
    :return:
    """
    feature_importance = np.load(feature_importance)
    y_pred = np.load(y_pred)
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, 'train')
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    class_list = get_class_list(data_dir, class_choice)
    model = FingerprintLSTM(input_dims, encoding_dims, seq_dims, class_list,
                            feature_choice, model_path, model_name, device, encoding)
    interpret_effectiveness(feature_importance, k, y_pred, dataset, model)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--stability', action='store_true', required=False, default=False)
    args.add_argument('--robustness', action='store_true', required=False, default=False)
    args.add_argument('--effectiveness', action='store_true', required=False, default=False)
    args.add_argument('--similar_model_importance_files', type=str, required=False)
    args.add_argument('--feature_importance', type=str, required=False)
    args.add_argument('--y_test', type=str, required=False)
    args.add_argument('--y_pred', type=str, required=False)
    args.add_argument('--k', type=int, required=True)
    args.add_argument('--data_dir', type=str, required=False)
    args.add_argument('--class_choice', type=str, required=False, default='all')
    args.add_argument('--scale', type=str, required=False)
    args.add_argument('--fields_file', type=str, required=False)
    args.add_argument('--input_dim', type=str, required=False, default='16', help='num or list of nums')
    args.add_argument('--encoding_dim', type=str, required=False, default='16', help='num or list of nums')
    args.add_argument('--seq_dim', type=str, required=False, default='32', help='num or list of nums')
    args.add_argument('--feature_choice', type=str, required=False, default=None,
                      help='name of choice file without suffix .joblib')
    args.add_argument('--model_path', type=str, required=False)
    args.add_argument('--model_name', type=str, required=False)
    args.add_argument('--no_encoding', action='store_true', required=False, default=False)
    args.add_argument('--gpu', type=str, required=False, default=None)
    args.add_argument('--subset', type=int, required=False, default=30)

    args = args.parse_args()
    print(args)

    args.input_dim = parse_str_dim(args.input_dim)
    args.encoding_dim = parse_str_dim(args.encoding_dim)
    args.seq_dim = parse_str_dim(args.seq_dim)

    if args.gpu is None:
        args.gpu = False
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.gpu = True

    if args.no_encoding:
        args.encoding = False
    else:
        args.encoding = True

    if args.stability:
        cal_stability(args.similar_model_importance_files, args.k)

    if args.robustness:
        cal_robustness(args.feature_importance, args.y_pred, args.k,
                       args.data_dir, args.class_choice, args.y_test, args.subset)

    if args.effectiveness:
        cal_effectiveness(args.feature_importance, args.k, args.y_pred, args.data_dir, args.class_choice,
                          args.scale, args.fields_file, args.input_dim, args.encoding_dim, args.seq_dim,
                          args.model_path, args.model_name, args.gpu, args.feature_choice, args.encoding)
