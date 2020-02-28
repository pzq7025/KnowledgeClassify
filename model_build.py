import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models.word2vec import Word2Vec
from keras import models, Input, Model
from keras import layers
from keras.layers.recurrent import LSTM, GRU, RNN
from keras.layers import Dense, LSTM, RNN, Bidirectional, Dropout, Flatten, BatchNormalization, Embedding, Conv1D, MaxPooling1D, concatenate, Convolution1D, MaxPool1D
import matplotlib.pyplot as plt
from keras.models import Sequential
import jieba
from init_model.attention_model import AttentionM, AttentionMC
from keras.utils import plot_model

# 机器学习
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

data_dir_path = './data/'
model_dir_path = './model/'
input_dir_path = './input/'
output_dir_path = './output/'

# 加载自定义词典
jieba.load_userdict(input_dir_path + 'dictionary_computer.txt')

# 配置训练参数
epochs_cnn = 50  # 训练次数 _后面代表模型
epochs_text_cnn = 50  # 训练次数
epochs_bilstm_att = epochs_text_cnn  # 训练次数
loss_rate = 0.1  # 丢失率
batch_size = 64  # 每个batch的大小


def stop_word():
    # 获取停用词
    with open(input_dir_path + 'stop_word.txt', encoding='utf-8') as f:
        stop_list = [line.strip('\n') for line in f.readlines()]
    return stop_list


def cut_word(text):
    content = [i for i in jieba.lcut(text) if i not in stop_word()]
    return content


def cut_words(text):
    return ' '.join(jieba.lcut(text))


# def draw_p_r_line(precision, recall, name=None):
#     plt.title(name + 'Precision/Recall Curve')  # give plot a title
#     plt.xlabel('Recall')  # make axis labels
#     plt.ylabel('Precision')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.plot(precision, recall)
#     plt.show()


def evaluate(accuracy, predict, name=None):
    """
    模型评估
    :param name: 模型名字
    :param accuracy: 真实值
    :param predict: 预测值
    :return:
    """
    print(f"accuracy--{name}:{metrics.accuracy_score(accuracy, predict):.4f}")
    print(f"recall(macro)--{name}:{metrics.recall_score(accuracy, predict, average='macro'):.4f}")
    print(f"recall(weighted)--{name}:{metrics.recall_score(accuracy, predict, average='weighted'):.4f}")
    print(f"f1-score--{name}:{metrics.f1_score(accuracy, predict, average='weighted', labels=np.unique(predict)):.4f}")
    print(f"kappa_score--{name}:{metrics.cohen_kappa_score(accuracy, predict):.4f}")
    print(f"precision_score--{name}:{metrics.precision_score(accuracy, predict, average='weighted'):.4f}")


def show_acc(history, name=None):
    """ 绘制精度曲线 """
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.title(name + ' Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.title(name + ' Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def cnn_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab):
    name_model = 'cnn_model'
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50))  # 使用Embeeding层将每个词编码转换为词向量
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(loss_rate))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(loss_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    one_hot_labels = to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    one_hot_labels_ = to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    history = model.fit(x_train_padded_seqs, one_hot_labels, epochs=epochs_cnn, batch_size=batch_size)
    loss_and_metrics = model.evaluate(x_test_padded_seqs, one_hot_labels_, batch_size=batch_size)
    print(f"cnn_model--loss_and_metrics:{loss_and_metrics}")
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
    y_predict = list(map(int, y_predict))
    # 注明模型名
    show_acc(history, name_model)
    # 模型评估
    evaluate(y_test, y_predict, name_model)
    # 注意训练出好的模型  保存较好的结果
    model.save(model_dir_path + name_model + '.h5')
    # plot_model(model, to_file=model_dir_path + 'cnn_model.png')


def text_cnn(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab):
    name_model = "text_cnn"
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(loss_rate)(flat)
    main_output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    history = model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=epochs_text_cnn)
    # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(int, result_labels))
    # 模型评估
    evaluate(list(y_test), y_predict, name=name_model)
    show_acc(history, name=name_model)
    # 注意训练出好的模型  所以最好每次都存入模型
    model.save(model_dir_path + name_model + '_model.h5')
    # plot_model(model, to_file=model_dir_path + 'text_cnn_model.png')


def bi_lstm(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab):
    # 输入
    name_model = "BILSTM"
    main_input = Input(shape=(50,), dtype='float64')
    embed = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)(main_input)
    # BILSTM+attention
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(embed)
    # attention = Attention(50, 64)(bi_lstm)
    flat = Flatten()(bi_lstm)
    drop = Dropout(loss_rate)(flat)
    main_output = Dense(3, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    history = model.fit(x_train_padded_seqs, one_hot_labels, batch_size=32, epochs=epochs_bilstm_att)
    # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(int, result_labels))
    # 模型评估
    evaluate(list(y_test), y_predict, name=name_model)
    show_acc(history, name=name_model)
    # 注意训练出好的模型
    model.save(model_dir_path + name_model + '.h5')
    # 图片保存
    # plot_model(model, to_file=model_dir_path + 'BILSTM.png')


def bi_lstm_attention(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab):
    # 输入
    name_model = "BILSTM_Attention"
    main_input = Input(shape=(50,), dtype='float64')
    embed = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)(main_input)
    # BILSTM+attention
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(embed)
    attention = AttentionMC()(bi_lstm)
    # flat = Flatten()(attention)
    drop = Dropout(loss_rate)(attention)
    main_output = Dense(3, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码epochs_bilstm_att
    history = model.fit(x_train_padded_seqs, one_hot_labels, batch_size=32, epochs=500)
    # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(int, result_labels))
    # 模型评估
    evaluate(list(y_test), y_predict, name=name_model)
    show_acc(history, name=name_model)
    # 注意训练出好的模型
    model.save(model_dir_path + name_model + '.h5')
    # 图片保存
    # plot_model(model, to_file=model_dir_path + name_model + '.png')


def rcnn_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab):
    # 输入
    name_model = "Rcnn"
    main_input = Input(shape=(50,), dtype='float64')
    embed = Embedding(len(vocab) + 1, 300, input_length=50)(main_input)
    cnn = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPool1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(embed)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn, rnn], axis=-1)
    main_output = Dense(3, activation='softmax')(con)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码epochs_bilstm_att
    history = model.fit(x_train_padded_seqs, one_hot_labels, batch_size=32, epochs=epochs_bilstm_att)
    # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(int, result_labels))
    # 模型评估
    evaluate(list(y_test), y_predict, name=name_model)
    show_acc(history, name=name_model)
    # 注意训练出好的模型
    model.save(model_dir_path + name_model + '.h5')
    # 图片保存
    # plot_model(model, to_file=model_dir_path + name_model + '.png')


def evaluate_machine(y_test, y_pred, name):
    print(f"accuracy--{name}:{metrics.accuracy_score(y_test, y_pred):.4f}")
    print(f"recall(macro)--{name}:{metrics.recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"recall(weighted)--{name}:{metrics.recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"f1-score--{name}:{metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)):.4f}")
    print(f"kappa_score--{name}:{metrics.cohen_kappa_score(y_test, y_pred):.4f}")
    print(f"precision_score--{name}:{metrics.precision_score(y_test, y_pred, average='weighted'):.4f}")


def nb_model(x_train, y_train, x_test, y_test, vector):
    # 输入
    name = "nb-model"
    nb = MultinomialNB()  # 贝叶斯
    pd.DataFrame(vector.fit_transform(x_train.cut_word).toarray(), columns=vector.get_feature_names())
    # print(pipe.steps)  # 查看pipeline的步骤（与pipeline相似）
    pipe = make_pipeline(vector, nb)
    right = cross_val_score(pipe, x_train.cut_word, y_train, cv=5, scoring='accuracy').mean()
    print(f"训练集准确率:{right}")
    # 拟合出模型
    model_fit = pipe.fit(x_train.cut_word, y_train)
    # print(model_type)

    # 测试数据
    pipe.predict(x_test.cut_word)

    # 保存预测结果
    y_pred = pipe.predict(x_test.cut_word)

    # 准确率 模型评估
    evaluate_machine(y_test, y_pred, name)


def svm_model(x_train, y_train, x_test, y_test, vector):
    # 输入
    name = "svm-model"
    svms = SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr', cache_size=500)
    pd.DataFrame(vector.fit_transform(x_train.cut_word).toarray(), columns=vector.get_feature_names())
    # print(pipe.steps)  # 查看pipeline的步骤（与pipeline相似）
    pipe = make_pipeline(vector, svms)
    right = cross_val_score(pipe, x_train.cut_word, y_train, cv=5, scoring='accuracy').mean()
    print(f"训练集准确率:{right}")
    # 拟合出模型
    model_fit = pipe.fit(x_train.cut_word, y_train)
    # print(model_type)

    # 测试数据
    pipe.predict(x_test.cut_word)

    # 保存预测结果
    y_pred = pipe.predict(x_test.cut_word)

    # 准确率 模型评估
    evaluate_machine(y_test, y_pred, name)


def random_forrest(x_train, y_train, x_test, y_test, vector):
    # 输入
    name = "random-forrest-model"
    random_forrestes = RandomForestClassifier(bootstrap=True, oob_score=True, criterion='gini')  # 贝叶斯
    pd.DataFrame(vector.fit_transform(x_train.cut_word).toarray(), columns=vector.get_feature_names())
    # print(pipe.steps)  # 查看pipeline的步骤（与pipeline相似）
    pipe = make_pipeline(vector, random_forrestes)
    right = cross_val_score(pipe, x_train.cut_word, y_train, cv=5, scoring='accuracy').mean()
    print(f"训练集准确率:{right}")
    # 拟合出模型
    model_fit = pipe.fit(x_train.cut_word, y_train)
    # print(model_type)

    # 测试数据
    pipe.predict(x_test.cut_word)

    # 保存预测结果
    y_pred = pipe.predict(x_test.cut_word)

    # 准确率 模型评估
    evaluate_machine(y_test, y_pred, name)


def data_detail_deep():
    # 获取数据  注意编码问题
    load_base = pd.read_csv(input_dir_path + "base.csv", encoding="GB2312")
    load_deep = pd.read_csv(input_dir_path + "deep.csv", encoding="GB2312")
    load_app = pd.read_csv(input_dir_path + "app.csv", encoding="GB2312")
    # 数据混合处理
    load_base['target'] = 0  # 基础知识标记为0
    load_deep['target'] = 1  # 基础知识标记为1
    load_app['target'] = 2  # 基础知识标记为2
    pf = pd.concat([load_base, load_deep, load_app], axis=0, ignore_index=True, join='outer')  # x为数据
    pf = pf.drop_duplicates()  # 去掉重复的评论
    pf = pf.dropna()  # 去除空数据
    # 进行分词
    pf['words'] = pf.text.apply(cut_word)

    tokenizer = Tokenizer()  # 创建一个Tokenizer对象
    # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    tokenizer.fit_on_texts(pf['words'])
    vocab = tokenizer.word_index  # 得到每个词的编号
    # print(f"vocab:{vocab}")
    # 数据划分
    x_train, x_test, y_train, y_test = train_test_split(pf['words'], pf['target'], test_size=0.2)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    # 序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  # 将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    # print(f"x_train_padded_seqs:{x_train_padded_seqs}")
    # print(f"x_train_padded_seqs shape:{x_train_padded_seqs.shape}")
    # print(f"x_test_padded_seqs:{x_test_padded_seqs}")
    # print(f"x_test_padded_seqs.shape:{x_test_padded_seqs.shape}")
    # print(f"y_train:{y_train}")
    # print(f"y_train shape:{y_train.shape}")
    # print(f"x_train:{x_train}")
    # print(f"x_train shape:{x_train.shape}")
    # CNN模型构建
    # cnn_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab)
    # text_cnn模型
    # text_cnn(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab)
    # bilstm模型
    # bi_lstm(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab)
    # BILSTM+attention模型
    bi_lstm_attention(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab)
    # cnn-RNN模型
    # rcnn_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab)


def traditional_detail():
    # 获取数据  注意编码问题
    load_base = pd.read_csv(input_dir_path + "base.csv", encoding="GB2312")
    load_deep = pd.read_csv(input_dir_path + "deep.csv", encoding="GB2312")
    load_app = pd.read_csv(input_dir_path + "app.csv", encoding="GB2312")
    # 数据混合处理
    load_base['target'] = 0  # 基础知识标记为0
    load_deep['target'] = 1  # 基础知识标记为1
    load_app['target'] = 2  # 基础知识标记为2
    pf = pd.concat([load_base, load_deep, load_app], axis=0, ignore_index=True, join='outer')  # x为数据
    pf = pf.drop_duplicates()  # 去掉重复的评论
    pf = pf.dropna()  # 去除空数据
    x = pd.concat([pf[['text']]])
    # x.columns = ['comments']
    y = pd.concat([pf.target])  # 标签
    x['cut_word'] = x.text.apply(cut_words)
    x_train, x_test, y_train, y_test = tts(x, y, random_state=42, test_size=0.2)  # 按八二的数据划分结果
    stop_words = stop_word()
    max_df = 0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    min_df = 3  # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
    vector = CountVectorizer(max_df=max_df,
                             min_df=min_df,
                             token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                             stop_words=frozenset(stop_words))  # 实例化 词袋模型

    # 贝叶斯算法
    # nb_model(x_train, y_train, x_test, y_test, vector)
    # SVM
    # svm_model(x_train, y_train, x_test, y_test, vector)
    # 随机森林
    # random_forrest(x_train, y_train, x_test, y_test, vector)


if __name__ == '__main__':
    # 深度学习
    data_detail_deep()
    # 传统的机器学习
    # traditional_detail()
