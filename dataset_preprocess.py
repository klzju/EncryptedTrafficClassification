import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import logging
from utils import check_path, get_fields_list

TIME_INTERVAL_INDEX = 1
PACKET_LEN_INDEX = 2

logging.basicConfig(

    level=logging.INFO,
    format='%(asctime)s-%(levelname)s %(message)s',
)


def merge_pcap_in_dir(pcap_dir, pcap_files, data_name):
    if os.path.exists(os.path.join(pcap_dir, data_name + '.pcap')):
        logging.warning('Pcap already exists.')
        return

    pcap_files = [os.path.join(pcap_dir, f) for f in pcap_files]
    in_files = ' '.join(pcap_files)
    print('&' * 50)
    # mergecap -w outfile infile1 infile2 ...
    cmd = 'mergecap -w {0}.pcap {1}'.format(os.path.join(pcap_dir, data_name), in_files)
    # print(cmd)
    ret = os.system(cmd)
    if ret == 0:
        logging.info('[DONE] Merge {0} pcap files.'.format(len(pcap_files)))
    else:
        logging.error('Merge pcap files.')
        exit(1)


def convert_pcap_to_csv(pcap_path, pcap_name, csv_dir, tshark_path, fields_file):
    csv_name = os.path.join(csv_dir, os.path.splitext(pcap_name)[0] + '.csv')
    if os.path.exists(csv_name):
        logging.warning('Csv already exists.')
        return
    with open(fields_file, 'r') as f:
        fields = f.readlines()
    fields = [line.strip('\n') for line in fields]
    fields = ['-e ' + line for line in fields]
    fields = ' '.join(fields)
    # tshark  -T fields $fields -r xxx.pcap -E header=y -E separator=, -E occurrence=f > xxx.csv
    # $fields = -e xxx -e xxx ...
    cmd_content = ' '.join([tshark_path, '-T', 'fields', fields, '-r', pcap_path, '-E', 'header=y',
                            '-E', 'occurrence=f', '-E', 'separator=,', '>', csv_name])
    ret = os.system(cmd_content)
    if ret == 0:
        logging.info('[DONE] Convert pcap to csv.')
    else:
        logging.error('Convert pcap to csv.')
        exit(1)


def packet_feature_extract(row, first_packet_tuple, prev_packet_time, fields):
    packet_tuple = '|'.join([str(row['ip.src']), str(row['ip.dst']),
                             str(row['tcp.srcport']), str(row['tcp.dstport']),
                             str(row['udp.srcport']), str(row['udp.dstport']),
                             ])

    fea = np.zeros(len(fields))
    # direction
    if packet_tuple != first_packet_tuple:
        fea[0] = 1
    else:
        fea[0] = 0

    fea[1] = (row['frame.time_epoch'] - prev_packet_time) * 1000  # ms
    if fea[1] > 86400 * 1000:
        return np.ones(len(fields)) * -1, False

    for i in range(2, len(fields)):  # feature fields need to iter start from 2, which is frame.len
        if 'version' in fields[i]:
            fea[2 + (i - 2)] = int(str(row[fields[i]]), 16)
        else:
            fea[2 + (i - 2)] = row[fields[i]]

    return fea, True


def session_feature_extract(data, idx, session, fields):
    first_packet_tuple = None
    prev_packet_time = 0
    for t in range(len(session)):
        if t >= 1024:
            logging.error('The num of session is above 1024')
            break
        packet = session.iloc[t]

        if t == 0:
            first_packet_tuple = '|'.join([str(packet['ip.src']), str(packet['ip.dst']),
                                           str(packet['tcp.srcport']), str(packet['tcp.dstport']),
                                           str(packet['udp.srcport']), str(packet['udp.dstport']),
                                           ])
            prev_packet_time = packet['frame.time_epoch']

        data[int(idx), t, :], ok = packet_feature_extract(packet, first_packet_tuple, prev_packet_time, fields)
        if not ok:
            return
        prev_packet_time = packet['frame.time_epoch']


def feature_extract(csv_dir, data_name, data_dir, fields_file):
    fields = get_fields_list(fields_file)
    df = pd.read_csv(os.path.join(csv_dir, data_name + '.csv'))
    df = df.fillna(-1)
    logging.info('Total {0} packets'.format(len(df)))
    df['tcp.stream'] = df['tcp.stream'].astype('int')
    df['udp.stream'] = df['udp.stream'].astype('int')
    df_tcp = df[df['tcp.stream'] != -1]
    df_udp = df[df['udp.stream'] != -1]

    max_seq_len = max(df_tcp.groupby('tcp.stream').size().max(), df_udp.groupby('udp.stream').size().max())
    mean_seq_len = max(df_tcp.groupby('tcp.stream').size().mean(), df_udp.groupby('udp.stream').size().mean())
    logging.info('max sequence length {0}'.format(max_seq_len))
    logging.info('mean sequence length {0}'.format(mean_seq_len))
    seq_len = min(1024, max_seq_len)  # ensure sequence length not be too large
    total_samples = df_tcp['tcp.stream'].unique().shape[0] + df_udp['udp.stream'].unique().shape[0]

    # -1 instead of 0 represents nothing
    data = np.ones((total_samples, seq_len, len(fields))) * -1

    for idx, session in df_tcp.groupby('tcp.stream'):
        session_feature_extract(data, idx, session, fields)
    tcp_session_num = len(df_tcp.groupby('tcp.stream'))
    for idx, session in df_udp.groupby('udp.stream'):
        session_feature_extract(data, idx + tcp_session_num, session, fields)
    logging.info('[INFO] Total {0} samples'.format(data.shape[0]))
    logging.info('[INFO] Max time interval in a session {0:.4f}'.format(data[:, :, TIME_INTERVAL_INDEX].max()))
    np.save(os.path.join(data_dir, data_name + '.npy'), data)


def gen_fsnet_dataset(data_dir):
    print('[INFO] generating fsnet dataset...')
    data_files = os.listdir(data_dir)
    data_files = [f for f in data_files if not f.startswith('.') and f.endswith('.npy') and 'scale' not in f]
    data_files = sorted(data_files)
    for idx, d in enumerate(data_files):
        data = np.load(os.path.join(data_dir, d))
        with open(os.path.join(data_dir, os.path.splitext(d)[0] + '.num'), 'w+') as f:
            for i in range(data.shape[0]):
                f.write(';')
                len_seq = data[i, :, PACKET_LEN_INDEX].tolist()
                len_seq = [str(int(a)) for a in len_seq]
                f.write('\t'.join(len_seq))
                f.write('\n')
        print('[DONE] [{0}/{1}] {2}'.format(idx + 1, len(data_files), d))


def check_args(args):
    logging.info("详细参数信息：")
    for arg, value in vars(args).items():
        logging.info(f"init args:{arg}-> {value}")
        # print(f"{arg}: {value}")
    if args.fsnet:
        gen_fsnet_dataset(args.data_dir)
        exit(0)
    if not os.path.exists(args.pcap_dir):
        print('[ERROR] pcap dir is not exist!')
        exit(1)
    if not os.path.exists(args.fields_file):
        print('[ERROR] fields file is not exist!')
        exit(1)
    check_path(args.csv_dir)
    check_path(args.data_dir)
    return args


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pcap_dir', type=str, required=False,
                      default='/media/kl/7c5ed3c9-49bd-46de-bbdd-976fbc893c6d/database/IDS-2017/pcap')
    args.add_argument('--csv_dir', type=str, required=False, default='./csv')
    args.add_argument('--tshark_path', type=str, required=False, default='/usr/bin/tshark')
    args.add_argument('--data_name', type=str, required=False, default='test')
    args.add_argument('--data_dir', type=str, required=False, default='./data')
    args.add_argument('--fields_file', type=str, required=False, default='./ids_fields.txt')
    args.add_argument('--fsnet', action='store_true', required=False, default=False)
    args.add_argument('--debug', type=bool, required=False, default=False)
    args = args.parse_args()
    check_args(args)
    pcap_files = sorted(os.listdir(args.pcap_dir))
    pcap_files = [f for f in pcap_files if (not f.startswith('.')) and (f.endswith('.pcap') or f.endswith('.pcapng'))]
    print('[INFO] Total {0} pcap files'.format(len(pcap_files)))

    merge_pcap_in_dir(args.pcap_dir, pcap_files, args.data_name)

    convert_pcap_to_csv(os.path.join(args.pcap_dir, args.data_name + '.pcap'), args.data_name + '.pcap',
                        args.csv_dir, args.tshark_path, args.fields_file)

    feature_extract(args.csv_dir, args.data_name, args.data_dir, args.fields_file)
