import copy
import os
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from datasets import TrafficDataset
from models import LSTM, FingerprintModule, MLPClassifier
from utils import get_class_list, get_fields_list


def train_knn(x_train, y_train, model, model_path, model_name, feature_choice=None):
    """
    :param x_train: ndarray
    :param y_train: ndarray
    :param model: KNeighborsClassifier()
    :param model_path: str
    :param model_name: str without suffix
    :param feature_choice: list of selected feature index
    :return: None
    """
    if feature_choice is not None:
        x_train = x_train[:, feature_choice]
    print('[INFO] training KNN...')
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print('[INFO] Train time (s): {0:.4f}'.format(end - start))
    joblib.dump(model, os.path.join(model_path, model_name + '.joblib'))


def test_knn(x_test, y_test, model, save_path, save_name, feature_choice=None):
    """
    :param x_test: ndarray
    :param y_test: ndarray
    :param model: KNeighborsClassifier()
    :param save_path: str
    :param save_name: str without suffix
    :param feature_choice: list of selected feature index
    :return: y_pred
    """
    if feature_choice is not None:
        x_test = x_test[:, feature_choice]
    print('[INFO] testing KNN...')
    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()
    print('[INFO] Accuracy: {0:.4f} Test time (s): {1:.4f}'.format(accuracy_score(y_test, y_pred), end - start))
    np.save(os.path.join(save_path, '_'.join([save_name, 'test']) + '.npy'), y_test)
    np.save(os.path.join(save_path, '_'.join([save_name, 'pred']) + '.npy'), y_pred)
    return y_pred


def knn_learning(data_dir, class_choice, scale, model_path, model_name, fields_file, seq_dim=32, feature_choice=None):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param model_path: str model save path
    :param model_name: str
    :param fields_file: str full path of fields file
    :param seq_dim: int
    :param feature_choice: str of dumped list of selected feature index .joblib
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='train')
    x_train, y_train = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    knn = KNeighborsClassifier()
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    train_knn(x_train, y_train, knn, model_path, model_name, feature_choice)


def knn_predict(data_dir, class_choice, scale, model_path, model_name,
                save_path, save_name, fields_file, seq_dim=32, feature_choice=None):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param model_path: str model save path
    :param model_name: str
    :param save_path: str result save path
    :param save_name: str
    :param fields_file: str full path of fields file
    :param seq_dim: int
    :param feature_choice: file of selected feature index .joblib
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='test')
    x_test, y_test = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    knn = joblib.load(os.path.join(model_path, model_name + '.joblib'))
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    y_pred = test_knn(x_test, y_test, knn, save_path, save_name, feature_choice)


def train_rf(x_train, y_train, model, model_path, model_name, feature_choice=None):
    """
    :param x_train: ndarray
    :param y_train: ndarray
    :param model: RandomForestClassifier()
    :param model_path: str
    :param model_name: str without suffix
    :param feature_choice: list of selected feature index
    :return: None
    """
    if feature_choice is not None:
        x_train = x_train[:, feature_choice]
    print('[INFO] training random forest...')
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print('[INFO] Train time (s): {0:.4f}'.format(end - start))
    joblib.dump(model, os.path.join(model_path, model_name + '.joblib'))


def test_rf(x_test, y_test, model, save_path, save_name, feature_choice=None):
    """
    :param x_test: ndarray
    :param y_test: ndarray
    :param model: RandomForestClassifier()
    :param save_path: str
    :param save_name: str without suffix
    :param feature_choice: list of selected feature index
    :return: y_pred
    """
    if feature_choice is not None:
        x_test = x_test[:, feature_choice]
    print('[INFO] testing random forest...')
    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()
    print('[INFO] Accuracy: {0:.4f} Test time (s): {1:.4f}'.format(accuracy_score(y_test, y_pred), end - start))
    np.save(os.path.join(save_path, '_'.join([save_name, 'test']) + '.npy'), y_test)
    np.save(os.path.join(save_path, '_'.join([save_name, 'pred']) + '.npy'), y_pred)
    return y_pred


def rf_learning(data_dir, class_choice, scale, model_path, model_name, fields_file, seq_dim=32, feature_choice=None):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param model_path: str model save path
    :param model_name: str
    :param fields_file: str full path of fields file
    :param seq_dim: int
    :param feature_choice: str of dumped list of selected feature index .joblib
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='train')
    x_train, y_train = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    rf = RandomForestClassifier(random_state=42)
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    train_rf(x_train, y_train, rf, model_path, model_name, feature_choice)


def rf_predict(data_dir, class_choice, scale, model_path, model_name,
               save_path, save_name, fields_file, seq_dim=32, feature_choice=None):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param model_path: str model save path
    :param model_name: str
    :param save_path: str result save path
    :param save_name: str
    :param fields_file: str full path of fields file
    :param seq_dim: int
    :param feature_choice: file of selected feature index .joblib
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='test')
    x_test, y_test = dataset.get_tabular_data(seq_dim), dataset.get_true_labels()
    rf = joblib.load(os.path.join(model_path, model_name + '.joblib'))
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    y_pred = test_rf(x_test, y_test, rf, save_path, save_name, feature_choice)


def train_mlp(n_epochs, dataloader, model, device, criterion, optimizer, model_path, model_name):
    print('[INFO] training MLP...')
    best_model_weight = None
    best_model_loss = None
    model.train()
    start = time.time()
    for e in range(n_epochs):
        epoch_loss = 0
        epoch_start = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out_batch = model(x_batch)
            loss = criterion(out_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('[INFO] Epoch [{0}/{1}] Iter [{2}/{3}] Loss: {4:.4f}'.format(
                    e + 1, n_epochs, batch_idx + 1, len(dataloader), loss.item()))
            epoch_loss += loss.item()
        epoch_end = time.time()
        if best_model_loss is None or epoch_loss < best_model_loss:
            best_model_loss = epoch_loss
            best_model_weight = copy.deepcopy(model.state_dict())
        print('[INFO] Epoch [{0}/{1}] Loss: {2:.4f} Time (s): {3:.4f}'.format(
            e + 1, n_epochs, epoch_loss, epoch_end - epoch_start))
    end = time.time()
    print('[INFO] Train time (s): {0:.4f} Best loss: {1:.4f}'.format(end - start, best_model_loss))
    torch.save(best_model_weight, os.path.join(model_path, model_name + '.pth'))


def test_mlp(dataloader, model, device, y_test, save_path, save_name):
    print('[INFO] testing MLP...')
    y_pred = torch.zeros(len(dataloader.dataset))
    p = 0  # index for samples
    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out_batch = model(x_batch)
            pred_batch = torch.argmax(out_batch, dim=1)
            y_pred[p:p + pred_batch.size(0)] = pred_batch[:]
            p += pred_batch.size(0)
    end = time.time()
    y_pred = y_pred.numpy()
    print('[INFO] Accuracy: {0:.4f} Test time (s): {1:.4f}'.format(
        accuracy_score(y_test, y_pred), end - start))
    np.save(os.path.join(save_path, '_'.join([save_name, 'test']) + '.npy'), y_test)
    np.save(os.path.join(save_path, '_'.join([save_name, 'pred']) + '.npy'), y_pred)
    return y_pred


def mlp_learning(data_dir, class_choice, scale, batch_size,
                 input_dim, seq_dim, hidden_dim, output_dim, model_path, model_name,
                 n_epochs, lr, fields_file, feature_choice=None, gpu=False):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param batch_size: int
    :param input_dim: not used the actual input dim is calculated
    :param seq_dim: int
    :param hidden_dim: int
    :param output_dim: int
    :param model_path: str
    :param model_name: str
    :param n_epochs: int
    :param lr: float
    :param fields_file: str full path of fields file
    :param feature_choice: file of selected feature index .joblib
    :param gpu: bool
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    input_dim = seq_dim * dataset.data.size(2)  # calculate the tabular input dim
    model = MLPClassifier(input_dim, hidden_dim, seq_dim, output_dim, feature_choice)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_mlp(n_epochs, dataloader, model, device, criterion, optimizer, model_path, model_name)


def mlp_predict(data_dir, class_choice, scale, batch_size,
                input_dim, seq_dim, hidden_dim, output_dim, model_path, model_name,
                save_path, save_name, fields_file, feature_choice=None, gpu=False):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param batch_size: int
    :param input_dim: not used the actual input dim is calculated
    :param seq_dim: int
    :param hidden_dim: int
    :param output_dim: int
    :param model_path: str
    :param model_name: str
    :param save_path: str result save path
    :param save_name: str
    :param fields_file: str full path of fields filemodels
    :param feature_choice: file of selected feature index .joblib
    :param gpu: bool
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    input_dim = seq_dim * dataset.data.size(2)  # calculate the tabular input dim
    model = MLPClassifier(input_dim, hidden_dim, seq_dim, output_dim, feature_choice)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name + '.pth')))
    y_pred = test_mlp(dataloader, model, device, dataset.get_true_labels(), save_path, save_name)


def train_lstm(n_epochs, dataloader, model, device, criterion, optimizer, model_path, model_name):
    print('[INFO] training lstm...')
    best_model_weight = None
    best_model_loss = None
    model.train()
    start = time.time()
    for e in range(n_epochs):
        epoch_loss = 0
        epoch_start = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out_batch = model(x_batch)
            loss = criterion(out_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('[INFO] Epoch [{0}/{1}] Iter [{2}/{3}] Loss: {4:.4f}'.format(
                    e + 1, n_epochs, batch_idx + 1, len(dataloader), loss.item()))
            epoch_loss += loss.item()
        epoch_end = time.time()
        if best_model_loss is None or epoch_loss < best_model_loss:
            best_model_loss = epoch_loss
            best_model_weight = copy.deepcopy(model.state_dict())
        print('[INFO] Epoch [{0}/{1}] Loss: {2:.4f} Time (s): {3:.4f}'.format(
            e + 1, n_epochs, epoch_loss, epoch_end - epoch_start))
    end = time.time()
    print('[INFO] Train time (s): {0:.4f} Best loss: {1:.4f}'.format(end - start, best_model_loss))
    torch.save(best_model_weight, os.path.join(model_path, model_name + '.pth'))


def test_lstm(dataloader, model, device, y_test, save_path, save_name):
    print('[INFO] testing lstm...')
    y_pred = torch.zeros(len(dataloader.dataset))
    p = 0  # index for samples
    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out_batch = model(x_batch)
            pred_batch = torch.argmax(out_batch, dim=1)
            y_pred[p:p + pred_batch.size(0)] = pred_batch[:]
            p += pred_batch.size(0)
    end = time.time()
    y_pred = y_pred.numpy()
    print('[INFO] Accuracy: {0:.4f} Test time (s): {1:.4f}'.format(
        accuracy_score(y_test, y_pred), end - start))
    np.save(os.path.join(save_path, '_'.join([save_name, 'test']) + '.npy'), y_test)
    np.save(os.path.join(save_path, '_'.join([save_name, 'pred']) + '.npy'), y_pred)
    return y_pred


def lstm_learning(data_dir, class_choice, scale, batch_size,
                  input_dim, hidden_dim, seq_dim, output_dim, model_path, model_name,
                  n_epochs, lr, fields_file, feature_choice=None, gpu=False, finetune=False):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param batch_size: int
    :param input_dim: not used the actual input dim is calculated
    :param hidden_dim: int
    :param seq_dim: int
    :param output_dim: int
    :param model_path: str
    :param model_name: str
    :param n_epochs: int
    :param lr: float
    :param fields_file: str full path of fields file
    :param feature_choice: file of selected feature index .joblib
    :param gpu: bool
    :param finetune: bool
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    model = LSTM(input_dim, hidden_dim, seq_dim, output_dim, feature_choice)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if finetune:
        model.load_state_dict(torch.load(os.path.join(model_path, model_name + '.pth')))
        model.fc = torch.nn.Linear(hidden_dim, output_dim + len(dataset.get_class_list()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_lstm(n_epochs, dataloader, model, device, criterion, optimizer, model_path, model_name)


def lstm_predict(data_dir, class_choice, scale, batch_size,
                 input_dim, hidden_dim, seq_dim, output_dim, model_path, model_name,
                 save_path, save_name, fields_file,
                 feature_choice=None, gpu=False, finetune=False, perturbation=0):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str name of scale factor file without suffix
    :param batch_size: int
    :param input_dim: not used the actual input dim is calculated
    :param hidden_dim: int
    :param seq_dim: int
    :param output_dim: int
    :param model_path: str
    :param model_name: str
    :param save_path: str result save path
    :param save_name: str
    :param fields_file: str full path of fields file
    :param feature_choice: file of selected feature index .joblib
    :param gpu: bool
    :param finetune: bool
    :param perturbation: float noise ratio added to the last two packets in a session
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode='test', perturbation=perturbation)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if feature_choice is not None:
        feature_choice = joblib.load(os.path.join(model_path, feature_choice + '.joblib'))
    if finetune:
        # get the output dim from class list
        # (--train --test situation the output_dim will be the original dim to load previous model)
        model = LSTM(input_dim, hidden_dim, seq_dim, len(dataset.get_class_list()), feature_choice)
    else:
        model = LSTM(input_dim, hidden_dim, seq_dim, output_dim, feature_choice)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name + '.pth')))
    y_pred = test_lstm(dataloader, model, device, dataset.get_true_labels(), save_path, save_name)


def train_fingerprint(n_epochs, dataloader, model, device, optimizer, model_path, model_name):
    print('[INFO] training fingerprint module...')
    best_model_weight = None
    best_model_loss = None
    model.train()
    start = time.time()
    for e in range(n_epochs):
        epoch_loss = 0
        epoch_start = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, loss, _, _ = model(x_batch)
            loss = loss.sum() / (x_batch.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('[INFO] Epoch [{0}/{1}] Iter [{2}/{3}] Loss: {4:.4f}'.format(
                    e + 1, n_epochs, batch_idx + 1, len(dataloader), loss.item()))
            epoch_loss += loss.item()
        epoch_end = time.time()
        if best_model_loss is None or epoch_loss < best_model_loss:
            best_model_loss = epoch_loss
            best_model_weight = copy.deepcopy(model.state_dict())
        print('[INFO] Epoch [{0}/{1}] Loss: {2:.4f} Time (s): {3:.4f}'.format(
            e + 1, n_epochs, epoch_loss, epoch_end - epoch_start))
    end = time.time()
    print('[INFO] Train time (s): {0:.4f} Best loss: {1:.4f}'.format(end - start, best_model_loss))
    torch.save(best_model_weight, os.path.join(model_path, model_name + '.pth'))
    return end - start


def test_fingerprint(dataloader, device, y_test, model_list, save_path, save_name, fields, seq_dims):
    print('[INFO] testing fingerprint module...')
    min_loss_all = torch.zeros(len(dataloader.dataset))
    all_loss_all = torch.zeros(len(dataloader.dataset), len(model_list))
    all_dim_loss_all = torch.zeros(len(dataloader.dataset), len(model_list), len(fields))
    all_dim_t_loss_all = torch.zeros(len(dataloader.dataset), len(model_list), max(seq_dims), len(fields))
    pred_all = torch.zeros(len(dataloader.dataset))
    p = 0  # index for samples
    start = time.time()
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            min_loss_batch = None
            all_loss_batch = torch.zeros(x_batch.size(0), len(model_list))
            all_dim_loss_batch = torch.zeros(x_batch.size(0), len(model_list), len(fields))
            all_dim_t_loss_batch = torch.zeros(x_batch.size(0), len(model_list), max(seq_dims), len(fields))
            pred_batch = torch.zeros(x_batch.size(0))
            for model_idx, model in enumerate(model_list):
                model.eval()
                _, loss, loss_dim, loss_dim_t = model(x_batch)
                all_loss_batch[:, model_idx] = loss
                all_dim_loss_batch[:, model_idx, :loss_dim.size(1)] = loss_dim
                all_dim_t_loss_batch[:, model_idx, :loss_dim_t.size(1), :loss_dim_t.size(2)] = loss_dim_t
                if min_loss_batch is None:
                    min_loss_batch = loss
                    pred_batch[:] = model_idx
                if (min_loss_batch > loss).any():
                    pred_batch[min_loss_batch > loss] = model_idx
                    min_loss_batch[min_loss_batch > loss] = loss[min_loss_batch > loss]
            min_loss_all[p:p + x_batch.size(0)] = min_loss_batch[:]
            pred_all[p:p + x_batch.size(0)] = pred_batch[:]
            all_loss_all[p:p + x_batch.size(0)] = all_loss_batch[:]
            all_dim_loss_all[p:p + x_batch.size(0)] = all_dim_loss_batch[:]
            all_dim_t_loss_all[p:p + x_batch.size(0)] = all_dim_t_loss_batch[:]
            p += x_batch.size(0)

    end = time.time()
    pred_all = pred_all.numpy()
    min_loss_all = min_loss_all.numpy()
    all_loss_all = all_loss_all.numpy()
    all_dim_loss_all = all_dim_loss_all.numpy()
    all_dim_t_loss_all = all_dim_t_loss_all.numpy()
    print('[INFO] Accuracy: {0:.4f} Test time (s): {1:.4f}'.format(accuracy_score(y_test, pred_all), end - start))
    np.save(os.path.join(save_path, '_'.join([save_name, 'test']) + '.npy'), y_test)
    np.save(os.path.join(save_path, '_'.join([save_name, 'pred']) + '.npy'), pred_all)
    np.save(os.path.join(save_path, '_'.join([save_name, 'min_loss_all']) + '.npy'), min_loss_all)
    np.save(os.path.join(save_path, '_'.join([save_name, 'all_loss_all']) + '.npy'), all_loss_all)
    np.save(os.path.join(save_path, '_'.join([save_name, 'all_dim_loss_all']) + '.npy'), all_dim_loss_all)
    np.save(os.path.join(save_path, '_'.join([save_name, 'all_dim_t_loss_all']) + '.npy'), all_dim_t_loss_all)
    return pred_all, min_loss_all, all_loss_all, all_dim_loss_all, all_dim_t_loss_all


def fingerprint_learning(logger, data_dir, class_choice, scale, batch_size, input_dims, encoding_dims, seq_dims,
                         model_path, model_name, n_epochs, lr, fields_file,
                         feature_choice=None, gpu=False, encoding=True, incremental=False):
    """
    :param logger:logs_printer
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str prefix of scale factor without suffix .npy
    :param batch_size: int
    :param input_dims: int or list of int
    :param encoding_dims: int or list of int
    :param seq_dims: int or list of int
    :param model_path: str
    :param model_name: str
    :param n_epochs: int or list of int
    :param lr: float
    :param fields_file: str full path of fields file
    :param feature_choice: str prefix of feature choice without suffix.joblib
    :param gpu: bool
    :param encoding: bool
    :param incremental: bool
    :return: None
    """
    class_choice = get_class_list(data_dir, class_choice)
    if incremental:
        trained_models = os.listdir(model_path)
        trained_models = [os.path.splitext(m)[0]
                          for m in trained_models if not m.startswith('.') and m.endswith('.pth')]
        trained_models = sorted(trained_models)
        need_to_train = []
        for c in class_choice:
            c_need = 1
            for m in trained_models:
                if c_need == 0:
                    break
                if c in m:
                    c_need = 0
            if c_need == 1:
                need_to_train.append(c)
        class_choice = need_to_train

    if type(input_dims) is not list:
        input_dims = [input_dims] * len(class_choice)
        encoding_dims = [encoding_dims] * len(class_choice)
        seq_dims = [seq_dims] * len(class_choice)
        n_epochs = [n_epochs] * len(class_choice)

    total_time = 0
    for idx, c in enumerate(class_choice):
        logger.info('training fingerprint module [{0}/{1}]'.format(idx + 1, len(class_choice)))
        dataset = TrafficDataset(data_dir, [c], scale, fields_file, 'train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if feature_choice is not None:
            tmp_feature_choice = joblib.load(os.path.join(model_path, '_'.join([feature_choice, c]) + '.joblib'))
        else:
            tmp_feature_choice = None
        model = FingerprintModule(input_dims[idx], encoding_dims[idx], seq_dims[idx], tmp_feature_choice, encoding)
        device = torch.device('cpu')
        if gpu:
            device = torch.device('cuda')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elapsed = train_fingerprint(n_epochs[idx], dataloader, model, device, optimizer,
                                    model_path, '_'.join([model_name, c]))
        total_time += elapsed
    logger.info('Total train time (s): {0:.4f}'.format(total_time))


def fingerprint_predict(data_dir, class_choice, scale, batch_size, input_dims, encoding_dims, seq_dims,
                        model_path, model_name, save_path, save_name,
                        fields_file, feature_choice=None, gpu=False, encoding=True, mode='test', perturbation=0):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :param scale: str prefix of scale factor without suffix .npy
    :param batch_size: int
    :param input_dims: int or list of int
    :param encoding_dims: int or list of int
    :param seq_dims: int or list of int
    :param model_path: str
    :param model_name: str
    :param save_path: str
    :param save_name: str
    :param fields_file: str full path of fields file
    :param feature_choice: str prefix of feature choice without suffix.joblib
    :param gpu: bool
    :param encoding: bool
    :param mode: use train data or test data
    :param perturbation: float noise ratio added to the last two packets in a session
    :return: None
    """
    dataset = TrafficDataset(data_dir, class_choice, scale, fields_file, mode, perturbation)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_choice = get_class_list(data_dir, class_choice)
    fields = get_fields_list(fields_file)
    model_list = []
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    if type(input_dims) is not list:
        input_dims = [input_dims] * len(class_choice)
        encoding_dims = [encoding_dims] * len(class_choice)
        seq_dims = [seq_dims] * len(class_choice)
    for idx, c in enumerate(class_choice):
        if feature_choice is not None:
            tmp_feature_choice = joblib.load(os.path.join(model_path, '_'.join([feature_choice, c]) + '.joblib'))
        else:
            tmp_feature_choice = None
        model = FingerprintModule(input_dims[idx], encoding_dims[idx], seq_dims[idx], tmp_feature_choice, encoding)
        model.load_state_dict(torch.load(os.path.join(model_path, '_'.join([model_name, c]) + '.pth')))
        model.to(device)
        model_list.append(model)
    pred_all, min_loss_all, all_loss_all, all_dim_loss_all, all_dim_t_loss_all = \
        test_fingerprint(dataloader, device, dataset.get_true_labels(), model_list, save_path, save_name,
                         fields, seq_dims)
