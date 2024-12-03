import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import check_path, heatmap, annotate_heatmap, get_fields_list, \
    horizontal_distribution_bar_chart


def loss_kde_fig(class_list, all_loss, y_true, save_path, save_name):
    """
    :param class_list: list of class names
    :param all_loss: ndarray samples x models
    :param y_true: ndarray
    :param save_path: str
    :param save_name: str without suffix
    :return:
    """
    # loss kernel density estimation plot
    for model_idx in range(len(class_list)):
        plt.figure(figsize=(8, 6), dpi=300)
        for class_idx in range(len(class_list)):
            loss = all_loss[y_true == class_idx][:, model_idx]
            sns.kdeplot(loss, label=class_list[class_idx], common_norm=True, fill=True)
        plt.grid()
        plt.xlabel('Loss', fontsize=30)
        plt.ylabel('Density', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.legend(fontsize=30, ncol=9, loc=(-1, -1))
        plt.savefig(os.path.join(save_path, '_'.join([save_name, class_list[model_idx]]) + '.pdf'),
                    bbox_inches='tight')
        plt.close()


def draw_loss_kde(all_loss, y_true, save_path, save_name):
    """
    :param all_loss: str for all loss file with suffix
    :param y_true: str for y_true file with suffix
    :param save_path: str
    :param save_name: str without suffix
    :return: None
    """
    class_list = ['Benign', 'Botnet', 'DDoS', 'DoS', 'FTP Patator',
                  'Port Scan', 'SSH Patator', 'Brute Force']
    all_loss = np.load(all_loss)
    y_true = np.load(y_true)
    loss_kde_fig(class_list, all_loss, y_true, save_path, save_name)


def perturbation_fig(perturbations, lstm, fingerprint, save_path, save_name, y_label):
    """
    :param perturbations: list of perturbation
    :param lstm: metric of lstm
    :param fingerprint: metric of fingerprint
    :param save_path: str
    :param save_name: str
    :param y_label: axis Y label
    :return:
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(perturbations, lstm, linestyle='--', linewidth=3, marker='*', markersize=13, label='Traditional LSTM')
    plt.plot(perturbations, fingerprint, linewidth=3, marker='o', markersize=13, label='Fingerprint LSTM')
    plt.grid()
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=25)
    plt.xlabel(r'$\beta$', fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')
    plt.close()


def draw_perturbation(save_path, save_name):
    perturbations = np.linspace(0, 1, 11)
    acc_lstm = [0.9649, 0.9649, 0.9649, 0.9645, 0.9639, 0.9629, 0.9528, 0.9526, 0.9522, 0.9515, 0.9426]
    acc_fingerprint = [0.9897, 0.9897, 0.9897, 0.9897, 0.9897, 0.9897, 0.9897, 0.9897, 0.9897, 0.9897, 0.9897]
    f1_lstm = [0.8124, 0.8123, 0.8120, 0.8117, 0.8113, 0.8110, 0.8106, 0.8072, 0.8035, 0.7881, 0.7597]
    f1_fingerprint = [0.9861, 0.9861, 0.9861, 0.9861, 0.9861, 0.9861, 0.9861, 0.9861, 0.9861, 0.9861, 0.9861]
    acc_lstm = [100 * a for a in acc_lstm]
    acc_fingerprint = [100 * a for a in acc_fingerprint]
    f1_lstm = [100 * a for a in f1_lstm]
    f1_fingerprint = [100 * a for a in f1_fingerprint]
    perturbation_fig(perturbations, acc_lstm, acc_fingerprint, save_path, '_'.join([save_name, 'acc']), 'Accuracy (%)')
    perturbation_fig(perturbations, f1_lstm, f1_fingerprint, save_path, '_'.join([save_name, 'f1']), 'F1-Score (%)')


def train_time_and_data_size_fig(incremental_size, lstm, fingerprint, save_path, save_name, y_label):
    """
    :param incremental_size: list of incremental size
    :param lstm: metric of lstm
    :param fingerprint: metric of fingerprint
    :param save_path: str
    :param save_name: str
    :param y_label: axis Y label
    :return:
    """
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(incremental_size, lstm, linestyle='--', linewidth=3, marker='*', markersize=13, label='Traditional LSTM')
    plt.plot(incremental_size, fingerprint, linewidth=3, marker='o', markersize=13, label='Fingerprint LSTM')
    plt.grid()
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=25)
    plt.xlabel(r'Incremental Data Num. ($\times 10^4$)', fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')
    plt.close()


def draw_train_time_and_data_size(save_path, save_name):
    inc_data_num = [2494, 5096, 7768, 10871, 13365, 15967, 18639, 20009, 20978, 25242]
    train_time_lstm = [79.88, 152.12, 395.62, 553.7, 991.24, 1217.8, 1504.86, 1790.44, 2074.82, 2439.66]
    train_time_fingerprint = [77.0981, 141.6117, 224.4963, 324.8296, 401.9277,
                              466.4413, 549.3259, 658.251, 763.7925, 896.7879]
    data_size_lstm = [545.5626221, 1660.31311, 3359.563843, 5737.595215, 8661.189209,
                      12153.97107, 16231.25317, 20608.22314, 25197.16223, 30718.85144]
    data_size_fingerprint = [545.5626221, 1114.750488, 1699.250732, 2378.031372, 2923.593994,
                             3492.78186, 4077.282104, 4376.969971, 4588.939087, 5521.689209]

    inc_data_num = [t/10000 for t in inc_data_num]
    train_time_lstm = [t/1000 for t in train_time_lstm]
    train_time_fingerprint = [t/1000 for t in train_time_fingerprint]
    data_size_lstm = [t/1024 for t in data_size_lstm]
    data_size_fingerprint = [t/1024 for t in data_size_fingerprint]
    train_time_and_data_size_fig(inc_data_num, train_time_lstm, train_time_fingerprint,
                                 save_path, '_'.join([save_name, 'time']), r'Train Time ($\times 10^3$s)')
    train_time_and_data_size_fig(inc_data_num, data_size_lstm, data_size_fingerprint,
                                 save_path, '_'.join([save_name, 'size']), 'Preserved Data Size (GB)')


def feature_rank_fig(feature_rank, feature_list, class_list, save_path, save_name, annotate=True):
    """
    :param feature_rank: ndarray features x traffic types  feature rank
    :param feature_list: list of feature names
    :param class_list: list of class names
    :param save_path: str
    :param save_name: str
    :param annotate: bool
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    im, cbar = heatmap(feature_rank, class_list, feature_list, ax=ax,
                       cmap="magma", cbarlabel="Feature Ranking")
    if annotate:
        texts = annotate_heatmap(im, valfmt="{x:d}", size=13, textcolors=('white', 'black'))
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')


def draw_feature_rank(feature_rank_csv_dir, save_path, save_name):
    feature_list = get_fields_list('ids_fields.txt')
    class_list = ['Benign', 'Botnet', 'DDoS', 'DoS',
                  'FTP Patator', 'Port Scan', 'SSH Patator', 'Brute Force']
    feature_rank = np.zeros((len(feature_list), len(class_list)), dtype=np.int)

    feature_rank_files = os.listdir(feature_rank_csv_dir)
    feature_rank_files = [f for f in feature_rank_files if f.endswith('.csv') and not f.startswith('.')]
    feature_rank_files = sorted(feature_rank_files)
    for idx, f in enumerate(feature_rank_files):
        rank = pd.read_csv(os.path.join(feature_rank_csv_dir, f), header=None)
        rank = rank[0]
        for i in range(rank.shape[0]):
            feature_rank[feature_list.index(rank[i]), idx] = i + 1

    feature_list[1] = 'frame.interval'  # change name when plot
    feature_rank_fig(feature_rank, feature_list, class_list, save_path, save_name)

    # draw feature importance per timestep (result not well throw away)
    # feature_importance_files = os.listdir(feature_rank_csv_dir)
    # feature_importance_files = [f for f in feature_importance_files if f.endswith('.npy') and not f.startswith('.')]
    # feature_importance_files = sorted(feature_importance_files)
    # for f in feature_importance_files:
    #     fea_importance = np.load(os.path.join(feature_rank_csv_dir, f))
    #     fea_importance = fea_importance.T  # feature importance t shape is timesteps x features so need transpose
    #     feature_rank_fig(fea_importance, feature_list, np.arange(0, 32), save_path,
    #                      os.path.splitext(f)[0], annotate=False)


def feature_rank_time_fig(methods, rank_time, save_path, save_name):
    """
    :param methods: str list of method names
    :param rank_time: time list of methods
    :param save_path: str
    :param save_name: str
    :return:
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(methods, rank_time, width=0.5, zorder=10)  # use zorder to control the bar not be covered by grid
    plt.grid(zorder=0)
    plt.tick_params(labelsize=25)
    plt.xlabel('Methods', fontsize=25)
    plt.ylabel('Time Cost (s)', fontsize=25)
    plt.yscale('log')
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')


def draw_feature_rank_time(save_path, save_name):
    methods = ['MI', 'RFE', 'RF', r'I$^2$RNN']
    rank_time = [53.6770*2, 0, 2049.1566, 15.0242]
    feature_rank_time_fig(methods, rank_time, save_path, '_'.join([save_name, 'time']))


def feature_rank_group_fig(methods, total, group1, group2, save_path, save_name):
    """
    :param methods: str list of method names
    :param total: f1 metric of total features
    :param group1: f1 metric of group1 features
    :param group2: f1 metric of group2 features
    :param save_path: str
    :param save_name: str
    :return:
    """
    plt.figure(figsize=(8, 6), dpi=300)

    x = np.arange(len(methods))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width, total, width, label='All', zorder=10)
    ax.bar(x, group1, width, label='Group 1', zorder=10)
    ax.bar(x + width, group2, width, label='Group 2', zorder=10)

    ax.set_ylabel('F1-Score', fontsize=20)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x, methods)
    ax.set_xlabel('Method', fontsize=20)
    plt.grid(zorder=0)
    plt.tick_params(labelsize=20)
    ax.legend(fontsize=15, ncol=3)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')


def draw_feature_rank_group(save_path, save_name):
    methods = ['MI', 'RF', 'I$^2$RNN']
    total = [0.9861, 0.9861, 0.9861]
    group1 = [0.8976, 0.8154, 0.9766]
    group2 = [0.6713, 0.6300, 0.6646]
    feature_rank_group_fig(methods, total, group1, group2, save_path, '_'.join([save_name, 'group']))


def feature_rank_num_fig(feature_num, mi, rf, i2rnn, save_path, save_name, y_label):
    """
    :param feature_num: list of feature nums
    :param mi: metric for mi
    :param rf: metric for rf
    :param i2rnn: metric for i2rnn
    :param save_path: str
    :param save_name: str
    :param y_label: axis Y label
    :return:
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(feature_num, mi, linestyle='--', linewidth=3, marker='*', markersize=13, label='MI')
    plt.plot(feature_num, rf, linestyle='-.', linewidth=3, marker='^', markersize=13, label='RF')
    plt.plot(feature_num, i2rnn, linewidth=3, marker='o', markersize=13, label=r'I$^2$RNN')
    plt.grid()
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=25)
    plt.xlabel(r'Feature Num.', fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')
    plt.close()


def draw_feature_rank_num(save_path, save_name):
    feature_num = list(range(2, 17, 2))
    feature_num = [str(a) for a in feature_num]
    acc_mi = [0.6582, 0.6899, 0.7920, 0.8998, 0.9500, 0.9834, 0.9860, 0.9897]
    acc_rf = [0.6912, 0.6897, 0.7921, 0.8953, 0.8968, 0.9389, 0.9680, 0.9897]
    acc_i2rnn = [0.9373, 0.9686, 0.9781, 0.9850, 0.9852, 0.9847, 0.9870, 0.9897]
    f1_mi = [0.7494, 0.8287, 0.8376, 0.8976, 0.8705, 0.8582, 0.8646, 0.9861]
    f1_rf = [0.8332, 0.8300, 0.8121, 0.8154, 0.8704, 0.7524, 0.8740, 0.9861]
    f1_i2rnn = [0.9473, 0.9708, 0.9438, 0.9766, 0.9765, 0.9766, 0.9763, 0.9861]
    # acc_mi = [100 * a for a in acc_mi]
    # acc_rf = [100 * a for a in acc_rf]
    # acc_i2rnn = [100 * a for a in acc_i2rnn]
    # f1_mi = [100 * a for a in f1_mi]
    # f1_rf = [100 * a for a in f1_rf]
    # f1_i2rnn = [100 * a for a in f1_i2rnn]
    feature_rank_num_fig(feature_num, acc_mi, acc_rf, acc_i2rnn,
                         save_path, '_'.join([save_name, 'num_acc']), 'Accuracy')
    feature_rank_num_fig(feature_num, f1_mi, f1_rf, f1_i2rnn,
                         save_path, '_'.join([save_name, 'num_f1']), 'F1-Score')


def feature_importance_local_fig(results, category_names, save_path, save_name):
    horizontal_distribution_bar_chart(results, category_names)
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')


def draw_feature_importance_local(feature_importance_local, y_test, y_pred, save_path, save_name):
    fields = get_fields_list('ids_fields.txt')
    fields[1] = 'frame.interval'
    class_list = ['Benign', 'Botnet', 'DDoS', 'DoS',
                  'FTP Patator', 'Port Scan', 'SSH Patator', 'Brute Force']

    feature_importance_local = np.load(feature_importance_local)
    norm_factor = feature_importance_local.sum(axis=1)
    norm_factor = norm_factor.reshape((-1, 1))
    feature_importance_local /= norm_factor
    y_test = np.load(y_test)
    y_pred = np.load(y_pred)

    results = {}
    for i in range(8):
        c_idx = np.where((y_test == i) & (y_test == y_pred))[0]
        select_idx = c_idx[0]
        results['{0}\n#{1}'.format(class_list[i], select_idx)] = feature_importance_local[select_idx]

    feature_importance_local_fig(results, fields, save_path, save_name)


def feature_local_cmp_fig(feature_num, lime, shap, perm, fingerprint, y_label, save_path, save_name):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(feature_num, lime, linestyle='--', linewidth=3, marker='*', markersize=13, label='LIME')
    plt.plot(feature_num, shap, linestyle='-.', linewidth=3, marker='^', markersize=13, label='SHAP')
    plt.plot(feature_num, perm, linestyle=':', linewidth=3, marker='X', markersize=13, label='PERM')
    plt.plot(feature_num, fingerprint, linewidth=3, marker='o', markersize=13, label=r'I$^2$RNN')
    plt.grid()
    plt.legend(fontsize=25)
    plt.tick_params(labelsize=30)
    plt.xlabel(r'$k$', fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')
    plt.close()


def draw_feature_local_cmp(save_path, save_name):
    feature_num = list(np.arange(2, 17, 2))
    feature_num = [str(e) for e in feature_num]

    lime_stability = [0.1293, 0.2536, 0.3773, 0.5013, 0.6256, 0.7502, 0.8750, 1.0000]
    shap_stability = [0.2308, 0.3566, 0.4806, 0.6042, 0.7278, 0.7516, 0.8757, 1.0000]
    perm_stability = [0.3882, 0.5082, 0.5846, 0.6513, 0.7133, 0.7680, 0.7999, 1.0000]
    fingerprint_stability = [0.5039, 0.6022, 0.7780, 0.8425, 0.8408, 0.8941, 0.9176, 1.0000]
    feature_local_cmp_fig(feature_num, lime_stability, shap_stability, perm_stability, fingerprint_stability,
                          'Stability', save_path, '_'.join([save_name, 'stability']))

    lime_robustness = [0.0031, 0.0030, 0.0024, 0.0019, 0.0013, 0.0007, 0.0003, 0.0000]
    shap_robustness = [0.1046, 0.1048, 0.1040, 0.1031, 0.1021, 0.0012, 0.0004, 0.0000]
    perm_robustness = [0.4301, 0.3778, 0.3509, 0.2380, 0.1325, 0.0406, 0.0001, 0.0000]
    fingerprint_robustness = [0.7002, 0.4698, 0.4120, 0.3143, 0.1807, 0.0610, 0.0419, 0.0000]
    feature_local_cmp_fig(feature_num, lime_robustness, shap_robustness, perm_robustness, fingerprint_robustness,
                          'Robustness', save_path, '_'.join([save_name, 'robustness']))

    lime_effectiveness = [0.7801, 0.8031, 0.8329, 0.8654, 0.8938, 0.9175, 0.9246, 0.9652]
    shap_effectiveness = [0.7759, 0.7922, 0.8162, 0.8498, 0.8802, 0.9106, 0.9180, 0.9652]
    perm_effectiveness = [0.7862, 0.7941, 0.8018, 0.8018, 0.8101, 0.8977, 0.9316, 0.9652]
    fingerprint_effectiveness = [0.8866, 0.8866, 0.8889, 0.9427, 0.9598, 0.9646, 0.9652, 0.9652]
    feature_local_cmp_fig(feature_num, lime_effectiveness, shap_effectiveness,
                          perm_effectiveness, fingerprint_effectiveness,
                          'Effectiveness', save_path, '_'.join([save_name, 'effectiveness']))


def time_scalability_fig(model_num, seq_prediction_time, parallel_prediction_time, save_path, save_name):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(model_num, seq_prediction_time, linestyle='--', linewidth=3, marker='o', markersize=13)
    plt.plot(model_num, parallel_prediction_time, linewidth=3, marker='*', markersize=13)
    plt.grid()
    plt.tick_params(labelsize=30)
    plt.xlabel(r'Fingerprint Modules Num.', fontsize=30)
    plt.ylabel('Time Cost (s)', fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(save_path, save_name + '.pdf'), bbox_inches='tight')
    plt.close()


def draw_time_scalability(save_path, save_name):
    # 365.6166/165.6417/136784 826.4497/687.5381/319151
    # 37.9027/7580 110.6725/17662

    model_num = [1, 10, 100, 1000, 10000]
    parallel_prediction_time = [21.54, 215.4, 2154.27, 25895.26, 267294.86]
    seq_prediction_time = [21.54, 21.54, 233.06, 2245.28, 10099.51]
    time_scalability_fig(model_num, seq_prediction_time, parallel_prediction_time, save_path, save_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, required=False, default='.')
    args.add_argument('--class_choice', type=str, required=False, default='all')
    args.add_argument('--feature_rank_csv_dir', type=str, required=False)
    args.add_argument('--feature_importance_local', type=str, required=False)
    args.add_argument('--save_path', type=str, required=True)
    args.add_argument('--save_name', type=str, required=True)
    args.add_argument('--all_loss', type=str, required=False)
    args.add_argument('--y_true', type=str, required=False)
    args.add_argument('--y_pred', type=str, required=False)

    args.add_argument('--kde', action='store_true', required=False, default=False)
    args.add_argument('--perturbation', action='store_true', required=False, default=False)
    args.add_argument('--inc_time_size', action='store_true', required=False, default=False)
    args.add_argument('--fea_rank', action='store_true', required=False, default=False)
    args.add_argument('--fea_rank_cmp', action='store_true', required=False, default=False)
    args.add_argument('--fea_importance_local', action='store_true', required=False, default=False)
    args.add_argument('--fea_local_cmp', action='store_true', required=False, default=False)
    args.add_argument('--time_scalability', action='store_true', required=False, default=False)

    args = args.parse_args()
    print(args)

    check_path(args.save_path)

    if args.kde:
        draw_loss_kde(args.all_loss, args.y_true,
                      args.save_path, args.save_name)

    if args.perturbation:
        draw_perturbation(args.save_path, args.save_name)

    if args.inc_time_size:
        draw_train_time_and_data_size(args.save_path, args.save_name)

    if args.fea_rank:
        draw_feature_rank(args.feature_rank_csv_dir, args.save_path, args.save_name)

    if args.fea_rank_cmp:
        draw_feature_rank_time(args.save_path, args.save_name)
        draw_feature_rank_group(args.save_path, args.save_name)
        draw_feature_rank_num(args.save_path, args.save_name)

    if args.fea_importance_local:
        draw_feature_importance_local(args.feature_importance_local, args.y_true, args.y_pred,
                                      args.save_path, args.save_name)

    if args.fea_local_cmp:
        draw_feature_local_cmp(args.save_path, args.save_name)

    if args.time_scalability:
        draw_time_scalability(args.save_path, args.save_name)
