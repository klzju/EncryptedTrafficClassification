import os

import joblib
import torch
from torch import nn


class FingerprintModule(nn.Module):
    def __init__(self, input_dim, encoding_dim, seq_dim, feature_choice=None, encoding=True):
        """
        :param input_dim: int
        :param encoding_dim: int
        :param seq_dim: int
        :param feature_choice: list of selected feature index
        :param encoding: bool
        """
        super(FingerprintModule, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = self.input_dim
        self.seq_dim = seq_dim
        self.feature_choice = feature_choice
        self.encoding = encoding
        if feature_choice is not None:
            self.input_dim = len(feature_choice)
        if self.encoding:
            self.encoding_layer = nn.Linear(self.input_dim, self.encoding_dim)
            self.lstm_unit = nn.LSTMCell(self.encoding_dim, self.hidden_dim)
        else:
            self.lstm_unit = nn.LSTMCell(self.input_dim, self.hidden_dim)

    def forward(self, x):
        h_0, c_0 = torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)
        h_n, c_n = h_0, c_0
        loss = 0
        loss_dim = 0
        loss_dim_t = torch.zeros(x.size(0), self.seq_dim, self.hidden_dim)
        if self.feature_choice is not None:
            x = x[:, :, self.feature_choice]
        for i in range(self.seq_dim):
            if i >= x.size(1):
                xi = torch.ones_like(x[:, 0, :]) * -1
            else:
                xi = x[:, i, :]
            if self.encoding:
                tmp_x = self.encoding_layer(xi)  # 65.22
                tmp_x = torch.tanh(tmp_x)  # 65.70
                h_n, c_n = self.lstm_unit(tmp_x, (h_n, c_n))
            else:
                h_n, c_n = self.lstm_unit(xi, (h_n, c_n))
            if i < self.seq_dim - 1:
                if i+1 >= x.size(1):
                    xi1 = torch.ones_like(x[:, 0, :]) * -1
                else:
                    xi1 = x[:, i+1, :]
                if i == 0:
                    loss = torch.sqrt(torch.square(h_n - xi1).sum(dim=1))
                    loss_dim = torch.sqrt(torch.square(h_n - xi1))
                    loss_dim_t[:, i, :] = torch.sqrt(torch.square(h_n - xi1))
                else:
                    tmp = loss
                    loss = tmp + torch.sqrt(torch.square(h_n - xi1).sum(dim=1))
                    loss_dim = loss_dim + torch.sqrt(torch.square(h_n - xi1))
                    loss_dim_t[:, i, :] = torch.sqrt(torch.square(h_n - xi1))
        tmp = loss
        loss = tmp / (self.seq_dim - 1)
        loss_dim = loss_dim / (self.seq_dim - 1)
        return h_n, loss, loss_dim, loss_dim_t


class FingerprintLSTM(nn.Module):
    def __init__(self, input_dims, encoding_dims, seq_dims, class_choice, feature_choice, model_path, model_name,
                 device, encoding=True):
        """
        used for Captum interpretability only
        :param input_dims: int or list of int
        :param encoding_dims: int or list of int
        :param seq_dims: int or list of int
        :param class_choice: list of chosen classes (processed)
        :param feature_choice: str prefix of feature choice without suffix.joblib
        :param model_path: str
        :param model_name: str
        :param device: torch.device
        :param encoding: bool
        """
        super(FingerprintLSTM, self).__init__()

        if type(input_dims) is not list:
            input_dims = [input_dims] * len(class_choice)
            encoding_dims = [encoding_dims] * len(class_choice)
            seq_dims = [seq_dims] * len(class_choice)

        self.model_list = []
        for idx, c in enumerate(class_choice):
            if feature_choice is not None:
                tmp_feature_choice = joblib.load(os.path.join(model_path, '_'.join([feature_choice, c]) + '.joblib'))
            else:
                tmp_feature_choice = None
            model = FingerprintModule(input_dims[idx], encoding_dims[idx], seq_dims[idx], tmp_feature_choice, encoding)
            model.load_state_dict(torch.load(os.path.join(model_path, '_'.join([model_name, c]) + '.pth')))
            model.to(device)
            self.model_list.append(model)

    def forward(self, x):
        all_loss_batch = torch.zeros(x.size(0), len(self.model_list))
        for model_idx, model in enumerate(self.model_list):
            model.eval()
            _, loss, _, _ = model(x)
            all_loss_batch[:, model_idx] = loss

        return all_loss_batch * -1


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim, output_dim, feature_choice=None):
        """
        :param input_dim: int
        :param hidden_dim: int
        :param seq_dim: int
        :param output_dim: int
        :param feature_choice: list of selected feature index
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_dim = seq_dim
        self.output_dim = output_dim
        self.feature_choice = feature_choice
        if feature_choice is not None:
            self.input_dim = len(feature_choice)

        self.lstm_unit = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        h_0, c_0 = torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)
        h_n, c_n = h_0, c_0
        if self.feature_choice is not None:
            x = x[:, :, self.feature_choice]
        for i in range(self.seq_dim):
            if i >= x.size(1):
                xi = torch.ones_like(x[:, 0, :]) * -1
            else:
                xi = x[:, i, :]
            h_n, c_n = self.lstm_unit(xi, (h_n, c_n))
        out = self.fc(h_n)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim, output_dim, feature_choice=None):
        """
        :param input_dim: int
        :param hidden_dim: int
        :param seq_dim: int used only for select how many time steps to tabular data
        :param output_dim: int
        :param feature_choice: list of selected feature index
        """
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_dim = seq_dim
        self.output_dim = output_dim
        self.feature_choice = feature_choice
        if feature_choice is not None:
            self.input_dim = len(feature_choice)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = x[:, :self.seq_dim, :]
        x = x.view(x.size(0), -1)
        if self.feature_choice is not None:
            x = x[:, self.feature_choice]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
