import os
from email.policy import default

from utils import check_path, parse_str_dim


def prepare_process(args):
    all_preprocess = AllPreProcess(args=args)
    all_preprocess.class_choice()
    all_preprocess.path_check()
    all_preprocess.args_check()
    all_preprocess.convert2int()
    all_preprocess.check_gpu()

    all_preprocess.print_all_args()
    return all_preprocess.args


class AllPreProcess:
    def __init__(self, args):
        # 循环赋值
        self.args = args
        self.args.encoding = False

    # 输出所有参数，供检查
    def print_all_args(self):
        self.args.logger.info('*' * 50 + '详细参数信息' + '*' * 50)
        for arg, value in vars(self.args).items():
            self.args.logger.info(f"init args----------------------{arg}<- {value}" if self.args.debug else None)
        self.args.logger.info('*' * 50 + '详细参数结束' + '*' * 50)

    # 判定是否需要编码
    def encoding(self):
        if self.args.no_encoding:
            self.args.encoding = False
        else:
            self.args.encoding = True

    # 处理特征选择情况，是否全选和选择部分两类
    def class_choice(self):
        if self.args.class_choice != 'all':
            self.args.class_choice = self.args.class_choice.split(',')

    def path_check(self):
        check_path(self.args.model_path)
        check_path(self.args.save_path)

    def args_check(self):
        if self.args.test and self.args.save_name is None:
            self.args.logger.error('[ERROR] no save name in test mode!')
            exit(1)
        if (self.args.train and self.args.finetune) or (self.args.train and self.args.incremental) or (
                self.args.finetune and self.args.incremental):
            self.args.logger.error('[ERROR] only support choose one train mode!')
            exit(1)
        # convert all str args to int args

    def convert2int(self):
        self.args.input_dim = parse_str_dim(self.args.input_dim)
        self.args.encoding_dim = parse_str_dim(self.args.encoding_dim)
        self.args.seq_dim = parse_str_dim(self.args.seq_dim)
        self.args.n_epochs = parse_str_dim(self.args.n_epochs)

    def check_gpu(self):
        if self.args.gpu is None:
            self.args.gpu = False
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
            self.args.gpu = True
