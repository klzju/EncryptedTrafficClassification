# I2RNN-code

论文**I2RNN: An Incremental and Interpretable Recurrent Neural Network for Encrypted Traffic Classification**的代码仓库。

## 运行环境

- Python 3.8.5
- numpy 1.21.5
- pandas 1.2.4
- matplotlib 3.5.1
- torch 1.13.0
- scikit-learn 1.2.2
- seaborn 0.11.0
- joblib 1.2.0

推荐先安装Anaconda环境，然后再安装上述依赖库：

```shell
pip install -r requirements.txt
```

## 文件说明

- ./figs: 保存各种图，raw目录下是原始素材，src是一些Photoshop源文件，其他的则是最终的图
- dataset_gen.py: 从原始pcap数据集生成训练测试数据文件
- datasets.py: 数据类定义文件，自定义实现了PyTorch数据类
- draw_figs.py: 画图代码
- feature_choice_file_gen.py: 生成特征选择文件
- ids_fields.txt: 针对CIC-IDS2017数据集需要提取的特征列表
- interpret_metrics.py: 可解释性的定量评估指标计算
- interpretability.py: 进行特征排序选择、类别距离相似度计算
- main.py: 主程序入口
- metrics.py: 定义了各种评估指标计算方式
- models.py: 模型类定义文件
- train_test.py: 模型训练测试代码
- utils.py: 通用函数方法
- vpn_fields.txt: 针对ISCXVPN2016数据集需要提取的特征列表

# 运行方式

## 数据集生成

需要针对每一个类别的数据单独运行

```shell
python dataset_gen.py --pcap_dir=单个类别pcaps的目录路径 \
        --csv_dir=中间生成的csv文件保存目录路径 \
        --data_name=类别名称 \
        --data_dir=生成的数据保存目录路径 \
        --fields_file=使用ids_fields.txt或者vpn_fields.txt或者自定义从csv中需要提取的特征字段
```

kl test

```shell
python dataset_gen.py --pcap_dir=/media/kl/7c5ed3c9-49bd-46de-bbdd-976fbc893c6d/database/IDS-2017 \
                      --csv_dir=./csv \
                      --data_name=test \
                      --tshark_path='/usr/bin/tshark' \
                      --data_dir=./data \
                      --fields_file=./ids_fields.txt
```

## 模型训练

```shell
python main.py --data_dir=生成的数据保存目录路径（与数据集生成时一致） --scale=scale --fields_file=与数据集生成时一致 --fingerprint --train --model_path=模型保存目录路径 --model_name=模型名称 --save_path=运行结果保存目录路径 --save_name=保存名称前缀 --input_dim=每一个类别的输入特征数量列表（和fields_file里的特征数一致） --encoding_dim=每一个类别的嵌入维度列表  --seq_dim=每一个类别的序列长度列表 --n_epochs=每一个类别需要训练多少个epoch列表（有几个类别就有几个数字）
```

e.g.

```shell
python main.py --data_dir=./data/ \
               --scale=scale \
               --fields_file=./ids_fields.txt \
               --fingerprint \
               --train \
               --model_path=./models \
               --model_name=fingerprint_ids \
               --save_path=./results \
               --save_name=fingerprint_ids \
               --input_dim=16,16,16,16,16,16,16,16 \
               --encoding_dim=8,8,8,8,8,8,8,8  \
               --seq_dim=32,32,32,32,32,32,32,32 \
               --n_epochs=100,50,100,100,50,100,50,50
```

## 模型测试

```shell
python main.py --data_dir=生成的数据保存目录路径（与数据集生成时一致） --scale=scale --fields_file=与数据集生成时一致 --fingerprint --test --model_path=模型保存目录路径（与模型训练时一致） --model_name=模型名称（与模型训练时一致） --save_path=运行结果保存目录路径（与模型训练时一致） --save_name=保存名称前缀（与模型训练时一致） --input_dim=每一个类别的输入特征数量列表（和fields_file里的特征数一致） --encoding_dim=每一个类别的嵌入维度列表  --seq_dim=每一个类别的序列长度列表 --n_epochs=每一个类别需要训练多少个epoch列表（有几个类别就有几个数字）
```

e.g.

```shell
python main.py --data_dir=/mnt/wangxinban/szx/IDS2017/data/ 
               --scale=scale 
               --fields_file=/mnt/wangxinban/szx/code/ids_fields.txt 
               --fingerprint 
               --test 
               --model_path=/mnt/wangxinban/szx/code/models 
               --model_name=fingerprint_ids 
               --save_path=/mnt/wangxinban/szx/code/results 
               --save_name=fingerprint_ids 
               --input_dim=16,16,16,16,16,16,16,16 
               --encoding_dim=8,8,8,8,8,8,8,8  
               --seq_dim=32,32,32,32,32,32,32,32 
               --n_epochs=100,50,100,100,50,100,50,50
```

## 结果评估

```shell
python -u metrics.py --y_test=运行结果保存目录路径下的xxx_test.npy --y_pred=运行结果保存目录路径下的xxx_pred.npy --save_path=运行结果保存目录路径 --save_name=运行结果保存名称（不要和上面一样） --data_dir=生成的数据保存目录路径（与数据集生成时一致）
```

e.g.

```shell
python metrics.py --y_test=/mnt/wangxinban/szx/code/results/fingerprint_ids_test.npy --y_pred=/mnt/wangxinban/szx/code/results/fingerprint_ids_pred.npy --save_path=/mnt/wangxinban/szx/code/results/ --save_name=fingerprint_ids_metrics --data_dir=/mnt/wangxinban/szx/IDS2017/data/
```

## 其他模型

除了使用I$^2$RNN模型进行训练测试，本仓库也支持KNN、随机森林、MLP、LSTM几种模型的训练和测试。相应命令大同小异，只需要更换模型参数为--knn、--rf、--mlp、--lstm即可。

### KNN

e.g.

```shell
# 训练
python main.py --data_dir=/mnt/wangxinban/szx/IDS2017/data/ --scale=scale --fields_file=/mnt/wangxinban/szx/code/ids_fields.txt --knn --train --model_path=/mnt/wangxinban/szx/code/models --model_name=knn_ids --save_path=/mnt/wangxinban/szx/code/results --save_name=knn_ids --seq_dim=32

# 测试
python main.py --data_dir=/mnt/wangxinban/szx/IDS2017/data/ --scale=scale --fields_file=/mnt/wangxinban/szx/code/ids_fields.txt --knn --test --model_path=/mnt/wangxinban/szx/code/models --model_name=knn_ids --save_path=/mnt/wangxinban/szx/code/results --save_name=knn_ids --seq_dim=32

# 结果评估
python metrics.py --y_test=/mnt/wangxinban/szx/code/results/knn_ids_test.npy --y_pred=/mnt/wangxinban/szx/code/results/knn_ids_pred.npy --save_path=/mnt/wangxinban/szx/code/results/ --save_name=knn_ids_metrics --data_dir=/mnt/wangxinban/szx/IDS2017/data/
```

### 随机森林

e.g.

```shell
# 训练
python main.py --data_dir=/mnt/szx/IDS2017/data/ --scale=scale --fields_file=/mnt/szx/code/ids_fields.txt --rf --train --model_path=/mnt/szx/code/models --model_name=rf_ids --save_path=/mnt/szx/code/results --save_name=rf_ids --seq_dim=32

# 测试
python main.py --data_dir=/mnt/szx/IDS2017/data/ --scale=scale --fields_file=/mnt/szx/code/ids_fields.txt --rf --test --model_path=/mnt/szx/code/models --model_name=rf_ids --save_path=/mnt/szx/code/results --save_name=rf_ids --seq_dim=32

# 结果评估
python metrics.py --y_test=/mnt/szx/code/results/rf_ids_test.npy --y_pred=/mnt/szx/code/results/rf_ids_pred.npy --save_path=/mnt/szx/code/results/ --save_name=rf_ids_metrics --data_dir=/mnt/szx/IDS2017/data/
```

### MLP

e.g.

```shell
# 训练
python main.py --data_dir=/mnt/szx/IDS2017/data/ --scale=scale --mlp --train --input_dim=16 --seq_dim=32 --hidden_dim=8 --output_dim=8 --model_path=/mnt/szx/code/models --model_name=mlp_ids --n_epochs=50 --fields_file=/mnt/szx/code/ids_fields.txt

# 测试
python main.py --data_dir=/mnt/szx/IDS2017/data/ --scale=scale --mlp --test --input_dim=16 --seq_dim=32 --hidden_dim=8 --output_dim=8 --model_path=/mnt/szx/code/models --model_name=mlp_ids --save_path=/mnt/szx/code/results/ --save_name=mlp_ids --fields_file=/mnt/szx/code/ids_fields.txt

# 结果评估
python metrics.py --y_test=/mnt/szx/code/results/mlp_ids_test.npy --y_pred=/mnt/szx/code/results/mlp_ids_pred.npy --save_path=/mnt/szx/code/results/ --save_name=mlp_ids_metrics --data_dir=/mnt/szx/IDS2017/data/
```

### LSTM

e.g.

```shell
# 训练
python main.py --data_dir=/mnt/szx/IDS2017/data/ --scale=scale --fields_file=/mnt/szx/code/ids_fields.txt --lstm --train --model_path=/mnt/szx/code/models --model_name=lstm_ids --save_path=/mnt/szx/code/results --save_name=lstm_ids --input_dim=16 --hidden_dim=8 --seq_dim=32 --output_dim=8 --n_epochs=2

# 测试
python main.py --data_dir=/mnt/szx/IDS2017/data/ --scale=scale --fields_file=/mnt/szx/code/ids_fields.txt --lstm --test --model_path=/mnt/szx/code/models --model_name=lstm_ids --save_path=/mnt/szx/code/results --save_name=lstm_ids --input_dim=16 --hidden_dim=8 --seq_dim=32 --output_dim=8

# 结果评估
python metrics.py --y_test=/mnt/szx/code/results/lstm_ids_test.npy --y_pred=/mnt/szx/code/results/lstm_ids_pred.npy --save_path=/mnt/szx/code/results/ --save_name=lstm_ids_metrics --data_dir=/mnt/szx/IDS2017/data/
```

## 模型可解释性

```shell
python interpretability.py --fingerprint --data_dir=生成的数据保存目录路径（与数据集生成时一致） --scale=scale --save_path=可解释结果保存目录路径 --save_name=保存结果名称前缀 --fields_file=与数据集生成时一致 --all_dim_loss=模型测试结果中的xxx_all_dim_loss_all.npy --all_dim_t_loss=模型测试结果中的xxx_all_dim_t_loss_all.npy --y_test=模型测试结果中的xxx_test.npy --y_pred=模型测试结果中的xxx_pred.npy
```

e.g.

```shell
python interpretability.py --fingerprint --data_dir=/Downloads/data/ --scale=scale --save_path=/Downloads/results/ --save_name=fingerprint_ids_interpretability --fields_file=/Documents/GitHub/I2RNNv3/code/ids_fields.txt --all_dim_loss=/Downloads/fingerprint_ids_all_dim_loss_all.npy --all_dim_t_loss=/Downloads/fingerprint_ids_all_dim_t_loss_all.npy --y_test=/Downloads/fingerprint_ids_test.npy --y_pred=/Downloads/fingerprint_ids_pred.npy
```