#coding=utf-8

# https://github.com/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb

import os
import sys
import time
import random
import numpy as np
import argparse
import json
import copy
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.backends.cudnn
import codecs
from transformers import AutoModel, AutoTokenizer, BertTokenizer, XLNetTokenizer


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self,
                 dropout,
                 num_classes,
                 model_name, freeze=False):
        super(CustomModel, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        if freeze:
            for k, param in self.pretrained_model.named_parameters():
                param.requires_grad = False
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.pretrained_model.config.hidden_size, num_classes)
        # self._init_weights(self.fc)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.pretrained_model(**inputs)
        pooler_output = outputs[0]
        return pooler_output

    def forward(self, inputs):
        out = self.feature(inputs)
        out = out.mean(dim=1)
        # out = out.max(dim=1).values
        output = self.fc(self.fc_dropout(out))
        return output


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



class DatasetIterater(object):
    def __init__(self, inputs, targets, batch_size, device):
        self.batch_size = batch_size
        self.inputs = inputs
        self.targets = targets
        self.n_batches = len(targets) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(targets) - self.n_batches * batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, inputs, targets):
        x = {}
        for k, vs in inputs.items():
            x[k] = torch.LongTensor(vs).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches_inputs = {}
            for k, v in self.inputs.items():
                batches_inputs[k] = v[self.index * self.batch_size: len(self.targets)]
            batches_targets = self.targets[self.index * self.batch_size: len(self.targets)]
            self.index += 1
            batches = self._to_tensor(batches_inputs, batches_targets)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches_inputs = {}
            for k, v in self.inputs.items():
                batches_inputs[k] = v[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches_targets = self.targets[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches_inputs, batches_targets)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


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


if __name__ == '__main__':

    # # test shuffle_data
    # x = [[1, 2, 3], [4], [5, 6], [7], [8, 9, 10]]
    # y = [3, 0, 2, 4, 1]
    # x_np = np.array(x)
    # y_np = np.array(y)
    # a, b = shuffle_data(x, y)
    # print(a)
    # print(b)
    # a, b = shuffle_data(x_np, y_np)
    # print(a)
    # print(b)
    # a, b = shuffle_data(x_np, y)
    # print(a)
    # print(b)
    # exit(0)

    # # test build n-gram x_index
    # UNK, PAD = '<UNK>', '<PAD>'
    # sentences = [
    #     ['哈喽','大家','好','啊'],
    #     ['哈哈', '呵呵'],
    #     ['啊'],
    # ]
    # x_index, vocabs = build_x_index(sentences, [2,3], min_freq=1)
    # sentences = [
    #     ['哈喽', '好', '啊'],
    #     ['哈哈', '呵呵', '嗯嗯'],
    #     ['啊'],
    # ]
    # x_index_2, _ = build_x_index(sentences, [2,3], min_freq=1, vocabs=vocabs)
    # exit(0)

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
    parser.add_argument('--model_name', type=str)
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
    model_name = args.model_name
    dropout = args.dropout

    if not model_name:
        print('no model_name!')
        exit(0)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device={}'.format(device))

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

    tokenizer = None
    if 'clue_xlnet' in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
    elif 'clue_' in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_train_index = []
    for text in data_train:
        data_train_index.append(tokenizer(text)['input_ids'])

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

    data_train_inputs = tokenizer(data_train, add_special_tokens=True, max_length=max_sent_len, truncation=True,
                                  padding="max_length", return_offsets_mapping=False)
    data_dev_inputs = tokenizer(data_dev, add_special_tokens=True, max_length=max_sent_len, truncation=True,
                                padding="max_length", return_offsets_mapping=False)
    data_test_inputs = tokenizer(data_test, add_special_tokens=True, max_length=max_sent_len, truncation=True,
                                 padding="max_length", return_offsets_mapping=False)

    model = CustomModel(0.5, num_classes, model_name, freeze=freeze)
    print(model)
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    # exit(0)
    model = model.to(device)
    best_acc = 0
    early_stop_patience = 0
    start_epoch = 0

    # resume = False
    # if resume:
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     model.load_state_dict(checkpoint['model'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start_time = time.time()
        # trainset
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batch = 0
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        for batch_idx, (inputs, targets) in enumerate(
                DatasetIterater(data_train_inputs, y_train_index, batch_size=batch_size, device=device)
        ):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()

            train_batch = batch_idx + 1

            # # test
            # if batch_idx >= 2:
            #     break

        epoch_end_time = time.time()
        # devset
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        dev_batch = 0
        # for batch_idx, (inputs, targets) in enumerate(devloader):
        for batch_idx, (inputs, targets) in enumerate(
                DatasetIterater(data_dev_inputs, y_dev_index, batch_size=batch_size, device=device)
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            dev_loss += loss.item()
            _, predicted = outputs.max(1)
            dev_total += targets.size(0)
            dev_correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()

            dev_batch = batch_idx + 1

            # # dev
            # if batch_idx >= 2:
            #     break

        print('epoch: {}/{}, {}s, train loss={:.4f}, train acc={:.2f}%, dev loss={:.4f}, dev acc={:.2f}%'.format(
            epoch + 1, start_epoch + epochs, int(epoch_end_time - epoch_start_time),
            train_loss / train_batch, train_correct * 100.0 / train_total,
            dev_loss / dev_batch, dev_correct * 100.0 / dev_total
        ))

        # save best acc
        dev_acc = dev_correct * 1.0 / dev_total
        if dev_acc > best_acc:
            # print('saving...')
            # state = {
            #     'model': model.state_dict(),
            #     'acc': dev_acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            best_acc = dev_acc
            early_stop_patience = 0

            # testset
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_batch = 0
            # for batch_idx, (inputs, targets) in enumerate(testloader):
            for batch_idx, (inputs, targets) in enumerate(
                    DatasetIterater(data_test_inputs, y_test_index, batch_size=batch_size, device=device)
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()

                test_batch = batch_idx + 1

            print('saving, test loss={:.4f}, test acc={:.2f}%'.format(
                test_loss / test_batch, test_correct * 100.0 / test_total
            ))
        else:
            early_stop_patience += 1
            if early_stop_patience >= patience:
                break

        # scheduler.step()

    end_time = time.time()
    print('time used={:.1f}s'.format(end_time - start_time))
