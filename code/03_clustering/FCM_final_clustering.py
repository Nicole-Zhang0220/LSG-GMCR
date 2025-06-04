import pandas as pd
import numpy as np
from fcmeans import FCM
from sklearn.preprocessing import LabelEncoder

# 加载数据
df = pd.read_csv('modified_combined_file_likecounts_with_predictions.csv')

# 确保 like_counts 列为数值型
df['like_counts'] = pd.to_numeric(df['like_counts'], errors='coerce').fillna(0)

# 对 like_counts 列应用对数变换
df['like_counts'] = pd.to_numeric(df['like_counts'], errors='coerce').fillna(0)
df['like_counts_log'] = np.log1p(df['like_counts'])

# 对 predicted_label 列进行编码（如果它是分类数据）
label_encoder = LabelEncoder()
df['predicted_label'] = label_encoder.fit_transform(df['predicted_label'])

# 准备聚类的特征
features = df[['like_counts_log', 'predicted_label']].values

# FCM 聚类
fcm = FCM(n_clusters=5)
fcm.fit(features)

# 获取聚类结果
fcm_labels = fcm.u.argmax(axis=1)

# 将聚类结果添加到 DataFrame
df['cluster'] = fcm_labels

# 保存带有聚类结果的 DataFrame
df.to_csv('clustered_data_modified.csv', index=False)

