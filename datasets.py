import os

import joblib
import numpy as np
import torch
from utils import get_class_list, get_fields_list

TRAIN_RATIO = 0.7


class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, class_choice, scale, fields_file, mode='train', perturbation=0):
        """
        :param data_dir: str
        :param class_choice: str 'all' or list ['class a', 'class b']
        :param scale: str name of scale factor file without '.npy'
        :param fields_file: str full path of fields file
        :param mode: str 'train' 'test' 'all'
        :param perturbation: float
        """
        fields = get_fields_list(fields_file)

        self.class_choice = get_class_list(data_dir, class_choice)

        data_list = []
        max_seq_len = 0
        total_samples = 0
        scale_factor = np.ones(len(fields))
        for c in self.class_choice:
            tmp_data = np.load(os.path.join(data_dir, c + '.npy'))
            if mode == 'train':
                tmp_data = tmp_data[:int(tmp_data.shape[0] * TRAIN_RATIO)]

                # get the scale factor
                scale_factor = np.ones(len(fields))
                for i in range(len(fields)):
                    scale_factor[i] = max(tmp_data[:, :, i].max(), 1)
                np.save(os.path.join(data_dir, '_'.join([scale, c]) + '.npy'), scale_factor)

            elif mode == 'test':
                scale_factor = np.load(os.path.join(data_dir, '_'.join([scale, c]) + '.npy'))
                tmp_data = tmp_data[int(tmp_data.shape[0] * TRAIN_RATIO):]
            max_seq_len = max(max_seq_len, tmp_data.shape[1])
            total_samples += tmp_data.shape[0]
        
            for i in range(len(fields)):
                tmp_data[:, :, i] /= scale_factor[i]
            # perturbation
            if mode == 'test':
                tmp_data[:, 30:32, :] = (1 - perturbation) * tmp_data[:, 30:32, :] + \
                                        perturbation * torch.randn_like(torch.tensor(tmp_data[:, 30:32, :])).numpy()

            data_list.append(tmp_data)

        # merge the data   -1 instead of 0 represents nothing
        self.data = np.ones((total_samples, max_seq_len, len(fields))) * -1
        self.label = np.ones(total_samples)
        p = 0
        for idx, d in enumerate(data_list):
            self.data[p:p + d.shape[0], :d.shape[1], :] = d[:, :, :]
            self.label[p:p + d.shape[0]] *= idx
            p += d.shape[0]

        self.data = torch.from_numpy(self.data).float()
        self.label = torch.from_numpy(self.label).long()

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

    def __len__(self):
        return len(self.data)

    def get_true_labels(self):
        """
        :return: all labels in numpy
        """
        return self.label.numpy()

    def get_class_list(self):
        """
        :return: class list in list
        """
        return self.class_choice

    def save_class_list(self, save_path):
        joblib.dump(self.class_choice, os.path.join(save_path, 'class_list.joblib'))

    def get_tabular_data(self, seq_dim):
        """
        :return: tabular data in numpy
        """
        tmp_data = self.data[:, :seq_dim, :]
        tabular_data = tmp_data.numpy()
        tabular_data = tabular_data.reshape((tabular_data.shape[0], -1))
        return tabular_data
