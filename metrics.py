import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix, precision_score, \
    recall_score, f1_score
from utils import get_class_list, check_path


def get_tn_fp_fn_tp_from_cm(cm):
    """
    :param cm: i-th true j-th pred
    :return: tn, fp, fn, tp
    """
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    return tn, fp, fn, tp


def get_tn_fp_fn_tp_from_mcm(mcm, class_index):
    """
    :param mcm: class_num x 2 x 2
    :param class_index: int
    :return: tn, fp, fn, tp
    """
    cm = mcm[class_index]
    tn, fp, fn, tp = get_tn_fp_fn_tp_from_cm(cm)
    return tn, fp, fn, tp


def cal_acc(tn, fp, fn, tp):
    return (tn + tp) / (tn + fp + fn + tp)


def cal_pre(tn, fp, fn, tp):
    return tp / (tp + fp)


def cal_re(tn, fp, fn, tp):
    return tp / (tp + fn)


def cal_f1(tn, fp, fn, tp):
    pre = cal_pre(tn, fp, fn, tp)
    re = cal_re(tn, fp, fn, tp)
    return 2 * pre * re / (pre + re)


def metric_class(metric, y_test, y_pred, mcm=None):
    """
    :param metric: 'acc' 'pre' 're' 'f1'
    :param y_test: ndarray
    :param y_pred: ndarray
    :param mcm: multilabel_confusion_matrix  shape: class_num x 2 x 2
    :return: metric result list of each class
    """
    if mcm is None:
        mcm = multilabel_confusion_matrix(y_test, y_pred)

    cal_func = {
        'acc': cal_acc,
        'pre': cal_pre,
        're': cal_re,
        'f1': cal_f1,
    }

    class_num = mcm.shape[0]
    res = []
    for i in range(class_num):
        tn, fp, fn, tp = get_tn_fp_fn_tp_from_mcm(mcm, i)
        res.append(cal_func[metric](tn, fp, fn, tp))
    return res


def print_and_save_class(metric, res, class_list, save_path, save_name):
    """
    :param metric: str list of metrics
    :param res: list of metrics of each class
    :param class_list: list of class names
    :param save_path: str
    :param save_name: str
    :return: None
    """
    f = open(os.path.join(save_path, '_'.join([save_name, 'class'])+'.txt'), 'w+')
    for c_idx, c in enumerate(class_list):
        s = []
        for m_idx, m in enumerate(metric):
            s.append('{0}: {1:.4f}'.format(m, res[m_idx][c_idx]))
        out = '[INFO] {0} {1}'.format(c, ' '.join(s))
        print(out)
        f.write(out)
        f.write('\n')
    f.close()


def eval_class(y_test, y_pred, class_list, save_path, save_name):
    """
    :param y_test: ndarray
    :param y_pred: ndarray
    :param class_list: list of class names
    :param save_path: str
    :param save_name: str
    :return: None
    """
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    acc = metric_class('acc', y_test, y_pred, mcm)
    pre = metric_class('pre', y_test, y_pred, mcm)
    re = metric_class('re', y_test, y_pred, mcm)
    f1 = metric_class('f1', y_test, y_pred, mcm)
    print_and_save_class(['acc', 'pre', 're', 'f1'], [acc, pre, re, f1],
                         class_list, save_path, save_name)


def print_and_save_total(metric, res, save_path, save_name):
    """
    :param metric: str list of metrics
    :param res: list of metric results
    :param save_path: str
    :param save_name: str
    :return: None
    """
    f = open(os.path.join(save_path, '_'.join([save_name, 'total'])+'.txt'), 'w+')
    s = []
    for m_idx, m in enumerate(metric):
        s.append('{0}: {1:.4f}'.format(m, res[m_idx]))
    out = '[INFO] Total {0}'.format(' '.join(s))
    print(out)
    f.write(out)
    f.write('\n')
    f.close()


def eval_total(y_test, y_pred, save_path, save_name):
    """

    :param y_test: ndarray
    :param y_pred: ndarray
    :param save_path: str
    :param save_name: str
    :return: None
    """

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    re = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print_and_save_total(['acc', 'pre', 're', 'f1'], [acc, pre, re, f1], save_path, save_name)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    joblib.dump(cm, os.path.join(save_path, '_'.join([save_name, 'cm'])+'.joblib'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--y_test', type=str, required=True)
    args.add_argument('--y_pred', type=str, required=True)
    args.add_argument('--save_path', type=str, required=True)
    args.add_argument('--save_name', type=str, required=True)
    args.add_argument('--data_dir', type=str, required=True, default=None)
    args.add_argument('--class_choice', type=str, required=False, default='all',
                      help='specify class in prediction result')

    args = args.parse_args()

    y_test = np.load(args.y_test)
    y_pred = np.load(args.y_pred)

    check_path(args.save_path)

    eval_total(y_test, y_pred, args.save_path, args.save_name)
    class_list = get_class_list(args.data_dir, args.class_choice)
    eval_class(y_test, y_pred, class_list, args.save_path, args.save_name)
