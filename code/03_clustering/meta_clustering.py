import pandas as pd
import numpy as np
from fcmeans import FCM

# 加载数据
df = pd.read_csv('text_similarity_clustered-ST.csv')

# 提取聚类结果作为特征
features = df[['predicted_label', 'cluster_TSD', 'cluster_St']].values

# FCM 聚类
fcm = FCM(n_clusters=5)
fcm.fit(features)

# 获取聚类结果
fcm_labels = fcm.u.argmax(axis=1)

# 将聚类结果添加到 DataFrame
df['cluster_metaFCMDBSCAN'] = fcm_labels

# 保留 id 和 review 列
df = df[['id', 'cutword', 'like_counts', 'predicted_label', 'cluster_TSD', 'cluster_St', 'meta_FCMDBSCAN', 'cluster_metaFCMDBSCAN']]

# 保存带有元聚类结果的 DataFrame
df.to_csv('meta_clustered_result.csv', index=False)
