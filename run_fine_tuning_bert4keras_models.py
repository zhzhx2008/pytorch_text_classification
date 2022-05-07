#coding=utf-8

import argparse
import codecs
import copy
import json
import os
import random
import time
import warnings

import numpy as np
from bert4keras.backend import K
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Lambda, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed

warnings.filterwarnings("ignore")
set_gelu('tanh')  # 切换gelu版本


def shuffle_data(*arrays):
    if not arrays:
        return None
    if isinstance(arrays[0], list):
        list_len = len(arrays[0])
    else:
        list_len = arrays[0].shape[0]
    idx = [i for i in range(0, list_len)]
    idx_shuffle = copy.deepcopy(idx)
    np.random.shuffle(idx_shuffle)
    res = []
    for x in arrays:
        if isinstance(x, list):
            res.append(np.array(x, dtype=object)[idx_shuffle].tolist())
        else:
            res.append(x[idx_shuffle])
    return tuple(res)


def build_dataset(data_path):
    sentences = []
    labels = []
    with codecs.open(data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            json_loads = json.loads(line)
            sentence = json_loads['sentence']
            sentences.append(sentence)
            label = json_loads['label_desc']
            labels.append(label)
    return sentences, labels


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def cal_sent_len(sentences, max_len_required=0.9997):
    sent_len_count = {}
    for sentence in sentences:
        sent_len_count[len(sentence)] = sent_len_count.get(len(sentence), 0) + 1
    sentences_count = sum(list(sent_len_count.values()))
    sent_len_count_list = [(k, sent_len_count[k]) for k in sorted(sent_len_count.keys(), reverse=True)]
    for k, v in sent_len_count_list:
        print('{}\t{}'.format(k, v))
    rm_sentences_count = 0
    for i, (k, v) in enumerate(sent_len_count_list):
        rm_sentences_count += v
        if (sentences_count - rm_sentences_count) / sentences_count * 1.0 < max_len_required:
            return sent_len_count_list[i + 1][0]


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument("--gpu", default="", type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=10000, type=int)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--max_sent_len_ratio', default=0.99971, type=float)
parser.add_argument('--max_sent_len', type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--freeze', action='store_true')
parser.add_argument("--model_type", default="bert", type=str)
parser.add_argument("--config_path", default='', type=str)
parser.add_argument("--checkpoint_path", default='', type=str)
parser.add_argument("--vocab_path", default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)
args, _ = parser.parse_known_args()
print(args)
# exit(0)

seed = args.seed
gpu = args.gpu
batch_size = args.batch_size
epochs = args.epochs
patience = args.patience
max_sent_len_ratio = args.max_sent_len_ratio
max_sent_len = args.max_sent_len
learning_rate = args.learning_rate
freeze = args.freeze
model_type = args.model_type
config_path = args.config_path
checkpoint_path = args.checkpoint_path
vocab_path = args.vocab_path
dropout = args.dropout

if not model_type:
    print('no model_type!')
    exit(0)

random.seed(seed)
np.random.seed(seed)
set_random_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

start_time = time.time()
print("loading data...")
data_train = './data/tnews_public/train.json'
data_test = './data/tnews_public/dev.json'  # test.json has no label, using dev.json repalced.
sentences, labels = build_dataset(data_train)
data_train, data_dev, labels_train, labels_dev = train_test_split(
    sentences, labels, test_size=0.1, shuffle=True, stratify=labels
)
# shuffle train/dev
data_train, labels_train = shuffle_data(data_train, labels_train)
data_dev, labels_dev = shuffle_data(data_dev, labels_dev)
data_test, labels_test = build_dataset(data_test)

# # test
# data_train = data_train[:32*4+24]
# data_dev = data_dev[:32+24]
# data_test = data_test[:32+24]
# labels_train = labels_train[:32*4 + 24]
# labels_dev = labels_dev[:32 + 24]
# labels_test =labels_test[:32 + 24]

print('train size={}'.format(len(data_train)))
print('dev size={}'.format(len(data_dev)))
print('test size={}'.format(len(data_test)))

labels_index_dict = {}
index_labels_dict = {}
labels_train_set = set(labels_train)
num_classes = len(labels_train_set)
for label in labels_train_set:
    labels_index_dict[label] = len(labels_index_dict)
index_labels_dict = {v: k for k, v in labels_index_dict.items()}
print('labels_index_dict={}'.format(labels_index_dict))
print('index_labels_dict={}'.format(index_labels_dict))

y_train_index = [labels_index_dict[x] for x in labels_train]
y_dev_index = [labels_index_dict[x] for x in labels_dev]
y_test_index = [labels_index_dict[x] for x in labels_test]
y_train_index = to_categorical(y_train_index, num_classes)
y_dev_index = to_categorical(y_dev_index, num_classes)
y_test_index = to_categorical(y_test_index, num_classes)

data_train_text_label = [(x, y) for x,y in zip(data_train, y_train_index)]
data_dev_text_label = [(x, y) for x,y in zip(data_dev, y_dev_index)]
data_test_text_label = [(x, y) for x,y in zip(data_test, y_test_index)]

# 建立分词器
tokenizer = Tokenizer(args.vocab_path, do_lower_case=True)
data_train_index = []
for text in data_train:
    data_train_index.append(tokenizer.encode(text)[0])

max_sent_len = 0  # default max len
for sent_index in data_train_index:
    if len(sent_index) > max_sent_len:
        max_sent_len = len(sent_index)
print('max_sent_len={}'.format(max_sent_len))
if max_sent_len_ratio and 0 < max_sent_len_ratio < 1:
    max_sent_len = cal_sent_len(data_train_index, max_sent_len_ratio)
    print('max_sent_len={}'.format(max_sent_len))
if max_sent_len:
    max_sent_len = max_sent_len
    print('max_sent_len={}'.format(max_sent_len))
# exit(0)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        random = False
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=max_sent_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=model_type,
    return_keras_model=False,
)
output = Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(bert.model.output)
output = Dropout(dropout)(output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

for layer in model.layers:
    layer.trainable = False
# 或者使用如下方法冻结所有层
# model.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True


# 可训练层
print('trainable layers:')
for x in model.trainable_weights:
    print(x.name)
# 不可训练层
print('untrainable layers:')
for x in model.non_trainable_weights:
    print(x.name)
for layer in model.layers:
    print(layer, layer.trainable)

# # 派生为带分段线性学习率的优化器。
# # 其中name参数可选，但最好填入，以区分不同的派生优化器。
# AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),  # 用足够小的学习率
    # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
    #     1000: 1,
    #     2000: 0.1
    # }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(data_train_text_label, batch_size)
valid_generator = data_generator(data_dev_text_label, batch_size)
test_generator = data_generator(data_test_text_label, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, patience=0):
        self.best_val_acc = 0.
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        super(Evaluator, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        train_acc = evaluate(train_generator)
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
            self.wait = 0
            test_acc = evaluate(test_generator)
            print('test acc={:.2f}%'.format(test_acc))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        print('train acc={:.2f}%, dev acc={:.2f}%'.format(train_acc, val_acc))


evaluator = Evaluator()

model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    verbose=2,
    callbacks=[evaluator]
)

end_time = time.time()
print('time used={:.1f}s'.format(end_time - start_time))
