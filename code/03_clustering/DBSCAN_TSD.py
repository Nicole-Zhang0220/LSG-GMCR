import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from fcmeans import FCM
from sklearn.preprocessing import LabelEncoder

# 加载数据
data = pd.read_csv('meta_clustered_data.csv')

# 确保like_counts列为数值型
data['like_counts'] = pd.to_numeric(data['like_counts'], errors='coerce')

# 提取id和like_counts列
# 假设id列是数据的索引，我们将仅使用like_counts列进行聚类
like_counts = data['like_counts'].values.reshape(-1, 1)

# DBSCAN聚类
# eps和min_samples是示例值，需要根据您的数据进行调整
db = DBSCAN(eps=10, min_samples=2).fit(like_counts)

# 聚类结果
labels = db.labels_

# 将聚类标签添加到原始数据中
data['cluster_label'] = labels

# 将更新后的数据集保存到新文件
data.to_csv('meta_clustered_with_clusters.csv', index=False)

# 显示带有聚类标签的数据
print(data)
