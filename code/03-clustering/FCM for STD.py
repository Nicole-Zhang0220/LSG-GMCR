import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from fcmeans import FCM
from sklearn.metrics.pairwise import cosine_similarity

# 加载 Word2Vec 模型
model = KeyedVectors.load_word2vec_format('sgns.weibo.bigram-char', binary=False)

# 定义函数将文本转换为向量
def text_to_vector(text):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 加载数据
df = pd.read_csv('updated cluster.csv')
df['cutword'] = df['cutword'].astype(str)

# 文本向量化
df['vector'] = df['cutword'].apply(text_to_vector)

# 计算相似度矩阵
similarity_matrix = cosine_similarity(np.array(list(df['vector'])))

# 使用相似度矩阵进行 FCM 聚类
fcm = FCM(n_clusters=20)
fcm.fit(similarity_matrix)
fcm_labels = fcm.u.argmax(axis=1)

# 将聚类结果添加到 DataFrame
df['cluster_St'] = fcm_labels

# 保存带有聚类结果的 DataFrame
df.to_csv('text_similarity_clustered-ST.csv', index=False)


