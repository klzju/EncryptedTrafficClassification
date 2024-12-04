import numpy
import pandas as pd

# 读取 CSV 文件
file_path = '/media/kl/7c5ed3c9-49bd-46de-bbdd-976fbc893c6d/database/IDS-2017/csv/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # 替换为你的 CSV 文件路径
# data = pd.read_csv(file_path)

import csv

# 读取 CSV 文件
file_path = file_path  # 替换为你的 CSV 文件路径
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    data = []
    # 输出每一行
    print("CSV 数据：")
    for row in reader:
        data.append(row)
data = numpy.array(data)
zhengchang = 0
gongji = 0
for item in data:
    if item[-1] != 'BENIGN':
        zhengchang += 1
    else:
        gongji += 1
print(zhengchang, gongji)

# print(data[1])
