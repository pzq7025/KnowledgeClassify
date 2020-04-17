from keras.models import load_model
import jieba
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib

input_model = './model/cnn_model_7731_9082' + '.h5'
input_dir_path = "./input/"
model_dir_path = "./model/" + 'wordModel' + '.pkl'


def stop_word():
    # 获取停用词
    with open(input_dir_path + 'stop_word.txt', encoding='utf-8') as f:
        stop_list = [line.strip('\n') for line in f.readlines()]
    return stop_list


def cut_word(text):
    content = []
    for i in jieba.lcut(text):
        if i not in stop_word():
            result = re.sub(r"\W+", "", i)
            if result is not "":
                content.append(result)
    return [' '.join(content)]


def detail_data():
    sentence = input("请输入句子：")
    result = cut_word(sentence)
    tokenizer = joblib.load(model_dir_path)
    x_train_word_ids = tokenizer.texts_to_sequences(result)
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50, padding='post')
    predict_data(x_train_padded_seqs)


def predict_data(content):
    model = load_model(input_model)
    y = model.predict_classes(content)
    print(y)


if __name__ == '__main__':
    detail_data()
# 人工智能在医学领域做图像分析   1
# 人工智能是门学科  0
