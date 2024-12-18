import argparse
import os
import logging
from train_test import knn_learning, knn_predict, rf_learning, rf_predict, mlp_learning, mlp_predict, lstm_learning, \
    lstm_predict, fingerprint_learning, fingerprint_predict
from utils import check_path, parse_str_dim, get_logger
from prepare_process import prepare_process

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, required=False, default='./data')
    args.add_argument('--class_choice', type=str, required=False, default='all',
                      help='lstm finetune: train or train &test only input new type, only test input all\n'
                           'fingerprint incremental: input all types')
    args.add_argument('--scale', type=str, required=True, help='name of scale factor file without suffix .npy')
    args.add_argument('--fields_file', type=str, required=True, help='full path of fields file')
    args.add_argument('--knn', action='store_true', required=False, default=False)
    args.add_argument('--rf', action='store_true', required=False, default=False)
    args.add_argument('--mlp', action='store_true', required=False, default=False)
    args.add_argument('--lstm', action='store_true', required=False, default=False)
    args.add_argument('--fingerprint', action='store_true', required=False, default=False)
    args.add_argument('--train', action='store_true', required=False, default=False)
    args.add_argument('--test', action='store_true', required=False, default=False)
    args.add_argument('--finetune', action='store_true', required=False, default=False)
    args.add_argument('--incremental', action='store_true', required=False, default=False)
    args.add_argument('--model_path', type=str, required=True)
    args.add_argument('--model_name', type=str, required=True)
    args.add_argument('--save_path', type=str, required=False, default='.')
    args.add_argument('--save_name', type=str, required=False, default=None)
    args.add_argument('--feature_choice', type=str, required=False, default=None,
                      help='name of choice file without suffix .joblib')
    args.add_argument('--input_dim', type=str, required=False, default='16', help='num or list of nums')
    args.add_argument('--encoding_dim', type=str, required=False, default='16', help='num or list of nums')
    args.add_argument('--hidden_dim', type=int, required=False, default=3)
    args.add_argument('--seq_dim', type=str, required=False, default='32', help='num or list of nums')
    args.add_argument('--output_dim', type=int, required=False, default=3)
    args.add_argument('--n_epochs', type=str, required=False, default='10', help='num or list of nums')
    args.add_argument('--lr', type=float, required=False, default=1e-3)
    args.add_argument('--batch_size', type=int, required=False, default=256)
    args.add_argument('--gpu', type=str, required=False, default=None)
    args.add_argument('--no_encoding', action='store_true', required=False, default=False)
    args.add_argument('--perturbation', type=float, required=False, default=0)
    args.add_argument('--debug', type=bool, required=False, default=False)
    args = args.parse_args()
    # start the logger to output run information
    args.logger = get_logger('kl_test', level=logging.INFO)
    args = prepare_process(args)
    if args.knn:
        if args.train:
            knn_learning(args.data_dir, args.class_choice, args.scale,
                         args.model_path, args.model_name, args.fields_file, args.seq_dim, args.feature_choice)
        if args.test:
            knn_predict(args.data_dir, args.class_choice, args.scale, args.model_path, args.model_name,
                        args.save_path, args.save_name, args.fields_file, args.seq_dim, args.feature_choice)

    if args.rf:
        if args.train:
            rf_learning(args.data_dir, args.class_choice, args.scale,
                        args.model_path, args.model_name, args.fields_file, args.seq_dim, args.feature_choice)
        if args.test:
            rf_predict(args.data_dir, args.class_choice, args.scale, args.model_path, args.model_name,
                       args.save_path, args.save_name, args.fields_file, args.seq_dim, args.feature_choice)

    if args.mlp:
        if args.train:
            mlp_learning(args.data_dir, args.class_choice, args.scale, args.batch_size, args.input_dim,
                         args.seq_dim, args.hidden_dim, args.output_dim, args.model_path, args.model_name,
                         args.n_epochs, args.lr, args.fields_file, args.feature_choice, args.gpu)
        if args.test:
            mlp_predict(args.data_dir, args.class_choice, args.scale, args.batch_size, args.input_dim,
                        args.seq_dim, args.hidden_dim, args.output_dim, args.model_path, args.model_name,
                        args.save_path, args.save_name, args.fields_file, args.feature_choice, args.gpu)

    if args.lstm:
        if args.train or args.finetune:
            lstm_learning(args.data_dir, args.class_choice, args.scale, args.batch_size,
                          args.input_dim, args.hidden_dim, args.seq_dim, args.output_dim,
                          args.model_path, args.model_name, args.n_epochs, args.lr,
                          args.fields_file, args.feature_choice, args.gpu, args.finetune)
        if args.test:
            lstm_predict(args.data_dir, args.class_choice, args.scale, args.batch_size,
                         args.input_dim, args.hidden_dim, args.seq_dim, args.output_dim,
                         args.model_path, args.model_name, args.save_path, args.save_name,
                         args.fields_file, args.feature_choice, args.gpu, args.finetune, args.perturbation)

    if args.fingerprint:
        if args.train or args.incremental:
            fingerprint_learning(logger=args.logger, data_dir=args.data_dir, class_choice=args.class_choice,
                                 scale=args.scale, batch_size=args.batch_size,
                                 input_dims=args.input_dim, encoding_dims=args.encoding_dim, seq_dims=args.seq_dim,
                                 model_path=args.model_path, model_name=args.model_name, n_epochs=args.n_epochs,
                                 lr=args.lr,
                                 fields_file=args.fields_file, feature_choice=args.feature_choice, gpu=args.gpu,
                                 encoding=args.encoding, incremental=args.incremental
                                 )
        if args.test:
            # test in both train and test data
            fingerprint_predict(args.data_dir, args.class_choice, args.scale, args.batch_size,
                                args.input_dim, args.encoding_dim, args.seq_dim,
                                args.model_path, args.model_name, args.save_path, '_'.join([args.save_name, 'train']),
                                args.fields_file, args.feature_choice,
                                args.gpu, args.encoding, 'train', args.perturbation)
            fingerprint_predict(args.data_dir, args.class_choice, args.scale, args.batch_size,
                                args.input_dim, args.encoding_dim, args.seq_dim,
                                args.model_path, args.model_name, args.save_path, args.save_name,
                                args.fields_file, args.feature_choice,
                                args.gpu, args.encoding, perturbation=args.perturbation)
