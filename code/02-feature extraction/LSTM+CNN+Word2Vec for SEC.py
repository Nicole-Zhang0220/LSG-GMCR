'''
lstm+cnn+Word2Vec
'''
from datetime import datetime

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from sklearn import metrics
import seaborn as sns
from keras import Sequential
from keras.layers import LSTM, Embedding, Dropout, Dense,Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPool1D
from gensim.models import Word2Vec

import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------------

epochs_num = 50
num_label = 2
max_len = 80
embedding_dim = 50
batch_size = 256

lstm_size=64
conkernel_num=32
conkernel_size=5
pool_size_num=3
dropout_rate=0.3
kernel_num=64

# ----------------------------------------------- ----------------------------------------
data_df = pd.read_csv("twofenlei_data_fc_random.csv", encoding='utf-8')

data = data_df['cutword'].values
label = data_df['label'].values

trainval_data, test_data, trainval_label, test_label = train_test_split(data, label, test_size=0.2, random_state=1000)
train_data,val_data,train_label,val_label=train_test_split(trainval_data,trainval_label,test_size=0.15,random_state=50)


# -------------------------------------------------------------------------------------------------
train_data = [str(a) for a in train_data.tolist()]
val_data =[str(a) for a in val_data.tolist()]
test_data = [str(a) for a in test_data.tolist()]

le = LabelEncoder()
train_label = le.fit_transform(train_label).reshape(-1, 1)
val_label=le.transform(val_label).reshape(-1,1)
test_label = le.transform(test_label).reshape(-1, 1)
ohe = OneHotEncoder()
train_label = ohe.fit_transform(train_label).toarray()
val_label=ohe.transform(val_label).toarray()
test_label = ohe.transform(test_label).toarray()

# -------------------------------------------------------
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(train_data)
tokenizer.fit_on_texts(val_data)
tokenizer.fit_on_texts(test_data)
train_data_sq = tokenizer.texts_to_sequences(train_data)
val_data_sq=tokenizer.texts_to_sequences(val_data)
test_data_sq = tokenizer.texts_to_sequences(test_data)

vocab_size = len(tokenizer.word_index) + 1

from keras_preprocessing.sequence import pad_sequences

train_data_sq_pading = pad_sequences(train_data_sq, padding='post', maxlen=max_len)
val_data_sq_pading=pad_sequences(val_data_sq,padding='post',maxlen=max_len)
test_data_sq_pading = pad_sequences(test_data_sq, padding='post', maxlen=max_len)

all_data = train_data + val_data + test_data
sentences = [text.split() for text in all_data]

word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

def create_embedding_matrix(word2vec_model, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word][:embedding_dim]
    return embedding_matrix

embedding_matrix = create_embedding_matrix(word2vec_model, tokenizer.word_index, embedding_dim)
#
# ##--------------------------------------------------------------------------------------
def created_model(lstm_size, vocab_size, embedding_dim, max_len,conkernel_num, conkernel_size,pool_size_num ,
                  dropout_rate,kernel_num,num_label):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=max_len,
                        trainable=True))
    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(Convolution1D(conkernel_num, conkernel_size, padding='same', strides=1, activation='relu'))
    model.add(MaxPool1D(pool_size=pool_size_num))
    model.add(Flatten())
    model.add(Dense(kernel_num, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_label, activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    return model

param_grid = dict(lstm_size=[64],
                  vocab_size=[len(tokenizer.index_word) + 1],
                  embedding_dim=[embedding_dim],
                  max_len=[max_len],
                  conkernel_num=[32],
                  conkernel_size=[3,5],
                  pool_size_num=[3],
                  dropout_rate=[0.3,],
                  kernel_num=[64],
                  num_label=[2],
                  )

param_outputflie = './param_out.txt'

# # ------------------------------------------------------------------------------------
model = KerasClassifier(build_fn=created_model, epochs=epochs_num, batch_size=batch_size, verbose=True)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5)
grid_result = grid.fit(train_data_sq_pading, train_label)

test_accuracy = grid.score(test_data_sq_pading, test_label)

with open(param_outputflie, 'a') as f:
    s = ('best Accuracy:'
         '{:.4f}\n{}\n test accuracy: {:.4f}\n\n')
    output_string = s.format(
        grid_result.best_score_,
        grid_result.best_params_,
        test_accuracy
    )
    print(output_string)
    f.write(output_string)


vocab_size=len(tokenizer.index_word) + 1
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=max_len,
                        trainable=True))
model.add(LSTM(lstm_size, return_sequences=True))
model.add(Convolution1D(conkernel_num, conkernel_size, padding='same', strides=1, activation='relu'))
model.add(MaxPool1D(pool_size=pool_size_num))
model.add(Flatten())
model.add(Dense(kernel_num, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(num_label, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=["accuracy"])


print("model training..........")
#learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
learning_rates = [0.0005]

target_dir = r'E:\PyCharm\pythonProject'
target_dir_models = r'E:\PyCharm\pythonProject'
for learning_rate in learning_rates:
    model = created_model(lstm_size, vocab_size, embedding_dim, max_len, conkernel_num,
                          conkernel_size, pool_size_num, dropout_rate, kernel_num, num_label)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    # 训练模型
    history = model.fit(train_data_sq_pading, train_label, epochs=epochs_num,
                        validation_data=(val_data_sq_pading, val_label),
                        batch_size=batch_size, verbose=True)

    model.save("model_lstm_word2vec.h5")
    loss,accuracy=model.evaluate(train_data_sq_pading,train_label)
    print("train_acc= {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(val_data_sq_pading, val_label)
    print("val_acc= {:.4f}".format(accuracy))

    history_dict = model_fit.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # Save the model
    model.save(os.path.join(target_dir_models, f"model_lstm_word2vec_{learning_rate}.h5"))

    # Record results in a DataFrame
    results = pd.DataFrame({
        'Train Accuracy': train_accuracy,
        'Validation Accuracy': val_accuracy,
        'Train Loss': train_loss,
        'Validation Loss': val_loss
    })

    # Save results to a new Excel file for each learning rate
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    results.to_excel(os.path.join(target_dir, f"results_{learning_rate}_{epochs_num}_{current_time}.xlsx"), index=False)

    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')
    plt.plot(range(epochs_num), val_loss, label='val_loss')
    plt.title('Loss curve')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(target_dir, f"loss_curve_{learning_rate}_{current_time}.png"))  # Save loss curve
    plt.figure()
    plt.plot(range(epochs_num), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs_num), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.title("Accuracy curve")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    # plt.show()
    # plt.savefig(os.path.join(target_dir, f"results_{learning_rate}_{current_time}.png"))
    plt.savefig(os.path.join(target_dir, f"accuracy_curve_{learning_rate}_{current_time}.png"))  # Save accuracy curve


print("model prediction...........")
model_names = os.listdir(target_dir_models)
test_result_dir = r'E:\PyCharm\pythonProject'
for model_name in model_names
    model = load_model(os.path.join(target_dir_models, model_name))
    test_pre = model.predict(test_data_sq_pading)
    confm = metrics.confusion_matrix(np.argmax(test_label, axis=1), np.argmax(test_pre, axis=1))
    print(confm)
    Labname = ["0", "1"]
    print(metrics.classification_report(np.argmax(test_label, axis=1), np.argmax(test_pre, axis=1)))
    plt.figure(figsize=(8, 8))
    sns.heatmap(confm.T, square=True, annot=True,
                fmt='d', cbar=False, linewidths=.6,
                cmap="YlGnBu")
    plt.xlabel('True label', size=14)
    plt.ylabel('Predicted label', size=14)
    plt.xticks(np.arange(num_label) + 0.5, Labname, size=12)
    plt.yticks(np.arange(num_label) + 0.5, Labname, size=12)
    plt.title('word2vec+lstm')
    plt.savefig(os.path.join(test_result_dir, f'{model_name}.png'))
    loss, accuracy = model.evaluate(test_data_sq_pading, test_label)
    print("test_acc= {:.4f}".format(accuracy))
    file_name = f'{model_name[:-3]}_result.txt'
    with open(os.path.join(test_result_dir, file_name), 'w') as f:
        f.write(f'{model_name}::: test_accuracy: {accuracy}, test_loss: {loss}\n')
        f.write(str(confm))

