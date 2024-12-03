import argparse
import os

import joblib
import pandas as pd

from utils import get_fields_list

feature_list = get_fields_list('ids_fields.txt')
class_list = ['benign', 'botnet', 'ddos', 'dos',
              'ftp_patator', 'portscan', 'ssh_patator', 'web_bf']


def gen_feature_choice_by_k(feature_rank_csv_dir, save_path, save_name):
    """
    generate feature choice for different k
    :param feature_rank_csv_dir: str dir of feature ranking csv files
    :param save_path: str
    :param save_name: str
    :return:
    """
    feature_ranking_files = os.listdir(feature_rank_csv_dir)
    feature_ranking_files = [f for f in feature_ranking_files if f.endswith('.csv') and not f.startswith('.')]
    feature_ranking_files = sorted(feature_ranking_files)

    for idx, f in enumerate(feature_ranking_files):
        feature_ranking = pd.read_csv(os.path.join(feature_rank_csv_dir, f), header=None)
        for k in [2, 4, 6, 8, 10, 12, 14]:
            select_feature = feature_ranking[0][:k]
            feature_choice = []
            for i in range(len(select_feature)):
                feature_choice.append(feature_list.index(select_feature[i]))
            feature_choice = sorted(feature_choice)

            if len(feature_ranking_files) == 1:
                for c in class_list:
                    joblib.dump(feature_choice, os.path.join(save_path, '_'.join([save_name, str(k), c]) + '.joblib'))
            else:
                joblib.dump(feature_choice, os.path.join(save_path,
                                                         '_'.join([save_name, str(k), class_list[idx]]) + '.joblib'))


def gen_feature_choice_by_group(feature_rank_csv_dir, save_path, save_name):
    """
    generate feature choice to two half group
    :param feature_rank_csv_dir: str dir of feature ranking csv files
    :param save_path: str
    :param save_name: str
    :return:
    """
    feature_ranking_files = os.listdir(feature_rank_csv_dir)
    feature_ranking_files = [f for f in feature_ranking_files if f.endswith('.csv') and not f.startswith('.')]
    feature_ranking_files = sorted(feature_ranking_files)

    for idx, f in enumerate(feature_ranking_files):
        feature_ranking = pd.read_csv(os.path.join(feature_rank_csv_dir, f), header=None)
        first_group_select_feature = feature_ranking[0][:8]
        first_group_feature_choice = []
        second_group_select_feature = feature_ranking[0][8:]
        second_group_feature_choice = []
        for i in range(len(first_group_select_feature)):
            first_group_feature_choice.append(feature_list.index(first_group_select_feature[i]))
        first_group_feature_choice = sorted(first_group_feature_choice)
        for i in range(8, 8+len(second_group_select_feature)):  # index starts from 8 to 15
            second_group_feature_choice.append(feature_list.index(second_group_select_feature[i]))
        second_group_feature_choice = sorted(second_group_feature_choice)

        if len(feature_ranking_files) == 1:
            for c in class_list:
                joblib.dump(first_group_feature_choice,
                            os.path.join(save_path, '_'.join([save_name, 'group1', c]) + '.joblib'))
                joblib.dump(second_group_feature_choice,
                            os.path.join(save_path, '_'.join([save_name, 'group2', c]) + '.joblib'))
        else:
            joblib.dump(first_group_feature_choice,
                        os.path.join(save_path, '_'.join([save_name, 'group1', class_list[idx]]) + '.joblib'))
            joblib.dump(second_group_feature_choice,
                        os.path.join(save_path, '_'.join([save_name, 'group2', class_list[idx]]) + '.joblib'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--feature_rank_csv_dir', type=str, required=True)
    args.add_argument('--save_path', type=str, required=True)
    args.add_argument('--save_name', type=str, required=True)
    args = args.parse_args()
    print(args)

    # gen_feature_choice_by_k(args.feature_rank_csv_dir, args.save_path, args.save_name)
    gen_feature_choice_by_group(args.feature_rank_csv_dir, args.save_path, args.save_name)
