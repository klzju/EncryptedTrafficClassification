import argparse
import os
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE

from datasets import TrafficDataset
from utils import check_path, get_class_list, get_fields_list, parse_str_dim

from captum.attr import Lime, KernelShap, FeaturePermutation
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function

from models import FingerprintLSTM

from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.render import make_snapshot

from snapshot_selenium import snapshot


def feature_ranking(fea_importance, save_path, save_name, feature_list, k=None):
    """
    :param fea_importance: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param feature_list: list of feature names
    :param k: int
    :return: None
    """
    fea_ranking = []
    for i in range(fea_importance.shape[0]):
        fea_ranking.append((i, fea_importance[i]))
    fea_ranking = sorted(fea_ranking, key=lambda e: e[1], reverse=True)
    if k is not None:
        fea_ranking = fea_ranking[:k]

    feature_choice = []
    for e in fea_ranking:
        feature_choice.append(e[0])
    feature_choice = sorted(feature_choice)
    joblib.dump(feature_choice, os.path.join(save_path, save_name + '.joblib'))

    with open(os.path.join(save_path, save_name+'.csv'), 'w+') as f:
        for e in fea_ranking:
            f.write('{0},{1}\n'.format(feature_list[e[0]], e[1]))


def mutual_info_ranking(x_train, y_train, save_path, save_name, feature_list, k=None):
    """
    :param x_train: ndarray
    :param y_train: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param feature_list: list of feature names
    :param k: int
    :return: None
    """
    print('[INFO] mutual info feature ranking...')
    start = time.time()
    fea_importance = mutual_info_classif(x_train, y_train, random_state=42)
    fea_importance = fea_importance.reshape((-1, len(feature_list)))
    fea_importance = fea_importance.sum(axis=0)
    end = time.time()
    print('[INFO] Time cost (s): {0:.4f}'.format(end-start))

    feature_ranking(fea_importance, save_path, save_name, feature_list, k)


def feature_ranking_mi(data_dir, class_choice, scale, save_path, save_name, fields_file, seq_dim, k=None):
    """
    :param data_dir: str
    :param class_choice: list of chosen classes
    :param scale: str of scale factor file without suffix .npy
    :param save_path: str
    :param save_name: str without suffix
    :param fields_file: str full path of fields file
    :param seq_dim: int  only used for limit selected time steps so that limit features
    :param k: int
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, 'train')
    x_train, y_train = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    fields = get_fields_list(fields_file)
    mutual_info_ranking(x_train, y_train, save_path, save_name, fields, k)


def rfe_ranking(x_train, y_train, save_path, save_name, feature_list, k=None):
    """
    :param x_train: ndarray
    :param y_train: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param feature_list: list of feature names
    :param k: int
    :return: None
    """
    print('[INFO] rfe feature ranking...')
    start = time.time()
    estimator = RandomForestClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=k)
    selector = selector.fit(x_train, y_train)
    fea_importance = np.array(selector.ranking_)
    fea_importance = fea_importance.reshape((-1, len(feature_list)))
    fea_importance = fea_importance.sum(axis=0)
    # convert ranking to importance, the ranking is higher the value is smaller and the importance is bigger
    fea_importance = fea_importance.max() - fea_importance
    end = time.time()
    print('[INFO] Time cost (s): {0:.4f}'.format(end-start))

    feature_ranking(fea_importance, save_path, save_name, feature_list, k)


def feature_ranking_rfe(data_dir, class_choice, scale, save_path, save_name, fields_file, seq_dim, k=None):
    """
    :param data_dir: str
    :param class_choice: str list of class names
    :param scale: str of scale factor file name without suffix .npy
    :param save_path: str
    :param save_name: str without suffix
    :param fields_file: str full path of fields file
    :param seq_dim: int
    :param k: int
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, 'train')
    x_train, y_train = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    fields = get_fields_list(fields_file)
    rfe_ranking(x_train, y_train, save_path, save_name, fields, k)


def rf_ranking(x_train, y_train, save_path, save_name, feature_list, k=None):
    """
    :param x_train: ndarray
    :param y_train: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param feature_list: list of feature names
    :param k: int
    :return: None
    """
    print('[INFO] rf feature ranking...')
    start = time.time()
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    fea_importance = rf.feature_importances_
    fea_importance = fea_importance.reshape((-1, len(feature_list)))
    fea_importance = fea_importance.sum(axis=0)
    end = time.time()
    print('[INFO] Time cost (s): {0:.4f}'.format(end-start))

    feature_ranking(fea_importance, save_path, save_name, feature_list, k)


def feature_ranking_rf(data_dir, class_choice, scale, save_path, save_name, fields_file, seq_dim, k=None):
    """
    :param data_dir: str
    :param class_choice: str list of class names
    :param scale: str of scale factor name without suffix
    :param save_path: str
    :param save_name: str without suffix
    :param fields_file: str full path of fields file
    :param seq_dim: int
    :param k: int
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, 'train')
    x_train, y_train = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    fields = get_fields_list(fields_file)
    rf_ranking(x_train, y_train, save_path, save_name, fields, k)


def fingerprint_ranking(all_dim_loss, all_dim_t_loss, y_test, y_pred,
                        save_path, save_name, class_choice, feature_list, k=None):
    """
    :param all_dim_loss: ndarray samples x models x features
    :param all_dim_t_loss: ndarray samples x models x timesteps x features
    :param y_test: ndarray
    :param y_pred: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param class_choice: str list of class names
    :param feature_list: list of feature names
    :param k: int
    :return: None
    """
    print('[INFO] fingerprint feature ranking...')
    class_num = all_dim_loss.shape[1]
    total_time = 0
    for i in range(class_num):
        start = time.time()
        c_dim_loss = all_dim_loss[(y_test == y_pred) & (y_test == i), :, :]
        c_dim_t_loss = all_dim_t_loss[(y_test == y_pred) & (y_test == i), :, :, :]
        fea_importance = np.zeros(len(feature_list))
        fea_importance_t = np.zeros((all_dim_t_loss.shape[2], len(feature_list)))
        for p in range(c_dim_loss.shape[0]):
            ii = c_dim_loss[p, i, :]
            ii_t = c_dim_t_loss[p, i, :, :]
            for m in range(class_num):
                tmp = c_dim_loss[p, m, :] - ii
                fea_importance += 1 * (tmp > 0) * tmp  # a numpy way to calculate relu
                tmp_t = c_dim_t_loss[p, m, :, :] - ii_t
                fea_importance_t += 1 * (tmp_t > 0) * tmp
        fea_importance /= (class_num * c_dim_loss.shape[0])
        fea_importance_t /= (class_num * c_dim_t_loss.shape[0])
        end = time.time()
        total_time += end - start
        print('[INFO] Ranking [{0}/{1}] Time cost (s): {2:.4f}'.format(i+1, class_num, end-start))

        np.save(os.path.join(save_path, '_'.join([save_name, class_choice[i], 't_importance'])+'.npy'),
                fea_importance_t)
        feature_ranking(fea_importance, save_path, '_'.join([save_name, class_choice[i]]), feature_list, k)

    print('[INFO] Total time (s): {0:.4f}'.format(total_time))


def feature_ranking_fingerprint(all_dim_loss, all_dim_t_loss, y_test, y_pred, save_path, save_name,
                                data_dir, class_choice, fields_file, k=None):
    """
    :param all_dim_loss: str file full path ndarray samples x models x features
    :param all_dim_t_loss: str file full path ndarray samples x models x timesteps x features
    :param y_test: str file full path ndarray
    :param y_pred: str file full path ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param data_dir: str
    :param class_choice: str list of class names
    :param fields_file: str full path of fields file
    :param k: int
    :return: None
    """
    all_dim_loss = np.load(all_dim_loss)
    all_dim_t_loss = np.load(all_dim_t_loss)
    y_test = np.load(y_test)
    y_pred = np.load(y_pred)
    class_choice = get_class_list(data_dir, class_choice)
    fields = get_fields_list(fields_file)
    fingerprint_ranking(all_dim_loss, all_dim_t_loss, y_test, y_pred, save_path, save_name, class_choice, fields, k)


def feature_importance(dataloader, model, interpreter, seq_dims, fields, save_path, save_name):
    """
    :param dataloader: torch.dataloader
    :param model: model with forward method
    :param interpreter: captum interpreter instance
    :param seq_dims: int or list of int
    :param fields: list of feature names
    :param save_path: str
    :param save_name: str without suffix
    :return:
    """
    if type(seq_dims) is not list:
        seq_dims = [seq_dims] * len(model.model_list)

    print('[INFO] Local feature importance calculating...')
    total_time = 0
    all_attr_list = []  # don't know the exact sample num
    p = 0  # to index sample
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        start = time.time()
        attr = interpreter.attribute(x_batch[:, :max(seq_dims), :], target=y_batch)
        all_attr_list.append(attr)
        p += x_batch.size(0)
        end = time.time()
        total_time += end - start
        print('[DONE] Calculated [{0}]/[{1}] Time Cost (s): {2:.4f}'.format(batch_idx+1, len(dataloader), end-start))

    # merge all samples' attr matrix
    all_attr_t = np.zeros((p, max(seq_dims), len(fields)))
    p = 0
    for a in all_attr_list:
        all_attr_t[p:p+a.size(0), :a.size(1), :] = a[:, :, :].numpy()
        p += a.size(0)
    all_attr = all_attr_t.sum(axis=1)
    np.save(os.path.join(save_path, save_name+'_attr_t.npy'), all_attr_t)
    np.save(os.path.join(save_path, save_name+'_attr.npy'), all_attr)
    print('[INFO] Total time: {0:.4f}'.format(total_time))


def feature_importance_local(method, data_dir, class_choice, scale, batch_size,
                             input_dims, encoding_dims, seq_dims, feature_choice,
                             model_path, model_name, save_path, save_name,
                             fields_file, encoding=True, gpu=False):
    """
    :param method: str  lime/shap/perm
    :param data_dir: str
    :param class_choice: list of chosen classes
    :param scale: str of scale factor file without suffix .npy
    :param batch_size: int
    :param input_dims: int or list of int
    :param encoding_dims: int or list of int
    :param seq_dims: int or list of int
    :param feature_choice: str prefix of feature choice without suffix.joblib
    :param model_path: str
    :param model_name: str without suffix
    :param save_path: str
    :param save_name: str without suffix
    :param fields_file: str full path of fields file
    :param encoding: bool
    :param gpu: bool
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, 'test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_choice = get_class_list(data_dir, class_choice)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    model = FingerprintLSTM(input_dims, encoding_dims, seq_dims, class_choice, feature_choice,
                            model_path, model_name, device, encoding)
    fields = get_fields_list(fields_file)

    if method == 'lime':
        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
        lime = Lime(model, interpretable_model=SkLearnLinearRegression(), similarity_func=exp_eucl_distance)
        print('[INFO] Starting calculating LIME feature importance...')
        feature_importance(dataloader, model, lime, seq_dims, fields, save_path, save_name+'_lime')
    elif method == 'shap':
        shap = KernelShap(model)
        print('[INFO] Starting calculating SHAP feature importance...')
        feature_importance(dataloader, model, shap, seq_dims, fields, save_path, save_name+'_shap')
    elif method == 'perm':
        perm = FeaturePermutation(model)
        print('[INFO] Starting calculating FeaturePermutation feature importance...')
        feature_importance(dataloader, model, perm, seq_dims, fields, save_path, save_name+'_perm')


def local_importance_fingerprint(all_dim_loss, y_test, save_path, save_name, feature_list):
    """
    :param all_dim_loss: ndarray samples x models x features
    :param y_test: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param feature_list: list of feature names
    :return: None
    """
    print('[INFO] Starting Calculating fingerprint local feature importance...')
    class_num = all_dim_loss.shape[1]
    all_attr = np.zeros((y_test.shape[0], len(feature_list)))
    start = time.time()
    for i in range(y_test.shape[0]):
        i_class = y_test[i]
        i_dim_loss = all_dim_loss[i, i_class]
        for j in range(class_num):
            tmp = all_dim_loss[i, j, :] - i_dim_loss
            tmp = 1 * (tmp > 0) * tmp
            all_attr[i, :] += tmp
        all_attr[i, :] /= class_num
    end = time.time()
    np.save(os.path.join(save_path, save_name+'_fingerprint_local.npy'), all_attr)
    print('[INFO] Total time (s): {0:.4f}'.format(end-start))


def fingerprint_local_importance(all_dim_loss, y_test, save_path, save_name, fields_file):
    """
    :param all_dim_loss: ndarray samples x models x features
    :param y_test: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param fields_file: str full path of fields file
    :return: None
    """
    all_dim_loss = np.load(all_dim_loss)
    y_test = np.load(y_test)
    fields = get_fields_list(fields_file)
    local_importance_fingerprint(all_dim_loss, y_test, save_path, save_name, fields)


def draw_distance_map(distance_map, class_choice, save_path, save_name, threshold=0):
    """
    :param distance_map: ndarray graph matrix
    :param class_choice: str list of chosen class names
    :param save_path: str
    :param save_name: str without suffix
    :param threshold: float minimum distance to draw
    :return: None
    """
    max_distance = distance_map.max()
    min_distance = distance_map[distance_map > 0].min()
    nodes = []
    for c in class_choice:
        nodes.append(opts.GraphNode(name=c, category=c, symbol_size=30, label_opts=opts.LabelOpts(font_size=40)))
    links = []
    for i in range(distance_map.shape[0]):
        for j in range(distance_map.shape[0]):
            if distance_map[i, j] > threshold:
                dis = '{0:.4f}'.format(distance_map[i, j])
                dis = float(dis)
                linewidth = 6 * (np.log(1+max_distance) - np.log(1+dis)) / np.log(1+max_distance)
                curve = 0.2
                if i > j:
                    curve = 0.3
                linecolor = None
                if abs(distance_map[i, j]-min_distance) < 1e-6:
                    linecolor = '#0000ff'
                elif abs(distance_map[i, j]-max_distance) < 1e-6:
                    linecolor = '#ff0000'
                links.append(opts.GraphLink(source=class_choice[i],
                                            target=class_choice[j],
                                            value=dis,
                                            symbol=[None, 'arrow'],
                                            linestyle_opts=opts.LineStyleOpts(
                                                width=max(1, linewidth),
                                                curve=curve,
                                                color=linecolor,
                                            )))

    categories = []
    for c in class_choice:
        categories.append({'name': c})  # format of categories are [{'name': xxx}, {'name': xxx}, ...]
    c = (
        Graph(init_opts=opts.InitOpts(width="1920px", height="1080px"))
        .add(
            "",
            nodes,
            links,
            layout='circular',
            categories=categories,
            repulsion=3000,
            # edge_label=opts.LabelOpts(
            #     is_show=True, position="middle", formatter="{c}"
            # ),
        )
        .set_global_opts(legend_opts=opts.LegendOpts(item_width=50, item_height=28,
                                                     textstyle_opts=opts.TextStyleOpts(font_size=40)))
    )

    make_snapshot(snapshot, c.render(path=os.path.join(save_path, save_name+'.html')),
                  os.path.join(save_path, save_name+'.pdf'), pixel_ratio=5, is_remove_html=False)


def cal_inter_class_distance(all_dim_loss, y_test, y_pred, save_path, save_name,
                             class_choice, feature_list, threshold=0):
    """
    :param all_dim_loss: ndarray samples x models x features
    :param y_test: ndarray
    :param y_pred: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param class_choice: str list of chosen class names
    :param feature_list: str list of feature names
    :param threshold: float minimum distance to draw
    :return: None
    """
    print('[INFO] calculating inter class distance...')
    class_num = all_dim_loss.shape[1]
    distance_map = np.zeros((class_num, class_num, len(feature_list)))
    for i in range(class_num):
        c_dim_loss = all_dim_loss[(y_test == y_pred) & (y_test == i), :, :]
        for p in range(c_dim_loss.shape[0]):
            ii = c_dim_loss[i, i, :]
            for m in range(class_num):
                tmp = c_dim_loss[i, m, :] - ii
                distance_map[i, m, :] += 1 * (tmp > 0) * tmp  # a numpy way to calculate relu

        distance_map[i, :, :] /= c_dim_loss.shape[0]

    distance_map_total = distance_map.sum(axis=2)
    np.save(os.path.join(save_path, '_'.join([save_name, 'dim'])+'.npy'), distance_map)
    np.save(os.path.join(save_path, '_'.join([save_name, 'total'])+'.npy'), distance_map_total)

    for k in range(len(feature_list)):
        draw_distance_map(distance_map[:, :, k], class_choice,
                          save_path, '_'.join([save_name, feature_list[k]]), threshold)
    draw_distance_map(distance_map_total, class_choice, save_path, '_'.join([save_name, 'total']), threshold)


def inter_class_distance(all_dim_loss, y_test, y_pred, save_path, save_name, data_dir,
                         class_choice, fields_file, threshold=0):
    """
    :param all_dim_loss: ndarray samples x models x features
    :param y_test: ndarray
    :param y_pred: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :param data_dir: str
    :param class_choice: str all or list of chosen class names
    :param fields_file: str full path of feature list file
    :param threshold: float minimum distance to draw
    :return:
    """
    all_dim_loss = np.load(all_dim_loss)
    y_test = np.load(y_test)
    y_pred = np.load(y_pred)
    class_choice = ['Benign', 'Botnet', 'DDoS', 'DoS', 'FTP Patator',
                    'Port Scan', 'SSH Patator', 'Brute Force']
    fields = get_fields_list(fields_file)
    cal_inter_class_distance(all_dim_loss, y_test, y_pred, save_path, save_name,
                             class_choice, fields, threshold)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mi', action='store_true', required=False, default=False)
    args.add_argument('--rfe', action='store_true', required=False, default=False)
    args.add_argument('--rf', action='store_true', required=False, default=False)
    args.add_argument('--fingerprint', action='store_true', required=False, default=False)
    args.add_argument('--lime', action='store_true', required=False, default=False)
    args.add_argument('--shap', action='store_true', required=False, default=False)
    args.add_argument('--perm', action='store_true', required=False, default=False)
    args.add_argument('--data_dir', type=str, required=True)
    args.add_argument('--class_choice', type=str, required=False, default='all')
    args.add_argument('--scale', type=str, required=True)
    args.add_argument('--save_path', type=str, required=True)
    args.add_argument('--save_name', type=str, required=True)
    args.add_argument('--fields_file', type=str, required=True)
    args.add_argument('--batch_size', type=int, required=False, default=32)
    args.add_argument('--input_dim', type=str, required=False, default='16', help='num or list of nums')
    args.add_argument('--encoding_dim', type=str, required=False, default='16', help='num or list of nums')
    args.add_argument('--seq_dim', type=str, required=False, default='32', help='num or list of nums')
    args.add_argument('--feature_choice', type=str, required=False, default=None,
                      help='name of choice file without suffix .joblib')
    args.add_argument('--model_path', type=str, required=False)
    args.add_argument('--model_name', type=str, required=False)
    args.add_argument('--all_dim_loss', type=str, required=False)
    args.add_argument('--all_dim_t_loss', type=str, required=False)
    args.add_argument('--y_test', type=str, required=False)
    args.add_argument('--y_pred', type=str, required=False)
    args.add_argument('--k', type=int, required=False, default=None)
    args.add_argument('--threshold', type=float, required=False, default=0)
    args.add_argument('--gpu', type=str, required=False, default=None)
    args.add_argument('--no_encoding', action='store_true', required=False, default=False)

    args = args.parse_args()
    print(args)

    check_path(args.save_path)

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

    if args.mi:
        feature_ranking_mi(args.data_dir, args.class_choice, args.scale,
                           args.save_path, args.save_name, args.fields_file, args.seq_dim, args.k)

    if args.rfe:
        feature_ranking_rfe(args.data_dir, args.class_choice, args.scale,
                            args.save_path, args.save_name, args.fields_file, args.seq_dim, args.k)

    if args.rf:
        feature_ranking_rf(args.data_dir, args.class_choice, args.scale,
                           args.save_path, args.save_name, args.fields_file, args.seq_dim, args.k)

    if args.fingerprint:
        feature_ranking_fingerprint(args.all_dim_loss, args.all_dim_t_loss, args.y_test, args.y_pred,
                                    args.save_path, args.save_name, args.data_dir,
                                    args.class_choice, args.fields_file, args.k)
        inter_class_distance(args.all_dim_loss, args.y_test, args.y_pred,
                             args.save_path, args.save_name, args.data_dir,
                             args.class_choice, args.fields_file, args.threshold)
        fingerprint_local_importance(args.all_dim_loss, args.y_test, args.save_path, args.save_name, args.fields_file)

    if args.lime:
        feature_importance_local('lime', args.data_dir, args.class_choice, args.scale, args.batch_size,
                                 args.input_dim, args.encoding_dim, args.seq_dim, args.feature_choice,
                                 args.model_path, args.model_name, args.save_path, args.save_name,
                                 args.fields_file, args.encoding, args.gpu)

    if args.shap:
        feature_importance_local('shap', args.data_dir, args.class_choice, args.scale, args.batch_size,
                                 args.input_dim, args.encoding_dim, args.seq_dim, args.feature_choice,
                                 args.model_path, args.model_name, args.save_path, args.save_name,
                                 args.fields_file, args.encoding, args.gpu)

    if args.perm:
        feature_importance_local('perm', args.data_dir, args.class_choice, args.scale, args.batch_size,
                                 args.input_dim, args.encoding_dim, args.seq_dim, args.feature_choice,
                                 args.model_path, args.model_name, args.save_path, args.save_name,
                                 args.fields_file, args.encoding, args.gpu)
