# encoding=utf-8

import os
import sys
import time
import random
import numpy as np
import argparse
import json
import jieba
import re
import copy
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.backends.cudnn
import codecs
from models.fasttext import FastTextModel
from models.textcnn1d import TextCNN1DModel
from models.textcnn2d import TextCNN2DModel
from models.textlstm import TextLSTMModel
from gensim.models import KeyedVectors


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


def build_dataset(data_path, num_char_no_split=False):
    sentences_word = []
    sentences_char = []
    labels = []
    with codecs.open(data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            json_loads = json.loads(line)
            sentence = json_loads['sentence']
            sentences_word.append(list(jieba.cut(sentence)))
            chars = ' '.join(list(sentence))
            if num_char_no_split:
                res = re.search(r'([0-9a-zA-Z]\s){2,}', chars)
                while res:
                    temp = res.group().strip()
                    chars = chars.replace(temp, temp.replace(' ', ''))
                    res = re.search(r'([0-9a-zA-Z]\s){2,}', chars)
            sentences_char.append(chars.split(' '))
            label = json_loads['label_desc']
            labels.append(label)
    return sentences_word, sentences_char, labels


def create_ngram(sent, ngram_value):
    return list(zip(*[sent[i:] for i in range(ngram_value)]))


def build_x_index(sentences, ngrams, vocabs=None, idx_start=0, max_size=None, min_freq=None):
    if not vocabs:
        vocabs = []
        # idx_start = 0
        for idx, n in enumerate(ngrams):
            vocab = {}
            for sentence in sentences:
                ngram_line_list = create_ngram(sentence, n)
                for ngram_word in ngram_line_list:
                    vocab[ngram_word] = vocab.get(ngram_word, 0) + 1
            min_freq_cur = 1
            if min_freq: min_freq_cur = min_freq[idx]
            vocab_list = sorted([_ for _ in vocab.items() if _[1] >= min_freq_cur], key=lambda x: x[1], reverse=True)
            if max_size and max_size[idx] < len(vocab_list):
                vocab_list = vocab_list[:max_size[idx]]
            vocab_dic = {word_count[0]: idx + idx_start for idx, word_count in enumerate(vocab_list)}
            vocab_dic.update({UNK: len(vocab_dic) + idx_start})
            vocabs.append(vocab_dic)
            idx_start += len(vocab_dic)
    x_index = []
    for sentence in sentences:
        x_index_line = []
        for idx, n in enumerate(ngrams):
            vocab = vocabs[idx]
            ngram_line_list = create_ngram(sentence, n)
            if not ngram_line_list:
                x_index_line.append(vocab.get(UNK))
            for ngram_word in ngram_line_list:
                x_index_line.append(vocab.get(ngram_word, vocab.get(UNK)))
        x_index.append(x_index_line)
    return x_index, vocabs


def row_concate(x1, x2):
    x = []
    for idx, x1_row in enumerate(x1):
        x_row = []
        x_row.extend(x1_row)
        x_row.extend(
            x2[idx]
        )
        x.append(x_row)
    return x


def sent_pad(x_index, max_sent_len, pad_value):
    x_index_new = []
    seq_lens = []
    for x in x_index:
        seq_len = len(x)
        if len(x) < max_sent_len:
            x.extend([pad_value] * (max_sent_len - len(x)))
        elif len(x) > max_sent_len:
            x = x[:max_sent_len]
            seq_len = max_sent_len
        x_index_new.append(x)
        seq_lens.append(seq_len)
    return x_index_new, seq_lens


class DatasetIterater(object):
    def __init__(self, inputs, targets, batch_size, device):
        self.batch_size = batch_size
        self.inputs = inputs[0]
        self.seq_lens = inputs[1]
        self.targets = targets
        self.n_batches = len(inputs[0]) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(inputs[0]) - self.n_batches * batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, inputs, seq_lens, targets):
        x = torch.LongTensor(inputs).to(self.device)
        x_seq_lens = torch.LongTensor(seq_lens).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        return (x, x_seq_lens), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches_inputs = self.inputs[self.index * self.batch_size: len(self.inputs)]
            batches_seq_lens = self.seq_lens[self.index * self.batch_size: len(self.seq_lens)]
            batches_targets = self.targets[self.index * self.batch_size: len(self.targets)]
            self.index += 1
            batches = self._to_tensor(batches_inputs, batches_seq_lens, batches_targets)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches_inputs = self.inputs[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches_seq_lens = self.seq_lens[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches_targets = self.targets[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches_inputs, batches_seq_lens, batches_targets)
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


def get_embedding_matrix_from_txt_file(embedding_file, word_index):
    def get_coefs(word, *arr):
        # if not isinstance(word, unicode):
        #     word = word.decode('utf-8')
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.rstrip().split(" ")) for o in codecs.open(embedding_file, 'r', encoding="utf-8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    print('all_embs.shape={}'.format(all_embs.shape))
    print('all_embs.mean={}'.format(emb_mean))
    print('all_embs.std={}'.format(emb_std))

    nb_words = len(word_index)
    print('nb_words={}'.format(nb_words))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i > nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count += 1
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                count += 1
            else:
                embedding_vector = embeddings_index.get(word.upper())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    count += 1
                else:
                    embedding_vector = embeddings_index.get(word.lower())
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        count += 1
    print('find embedding={}/{}'.format(count, nb_words))

    return embedding_matrix


def get_embedding_matrix_from_binary_file(embedding_file, word_index):
    word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

    emb_mean, emb_std = word2vec.vectors.mean(), word2vec.vectors.std()
    embed_size = word2vec.vectors.shape[1]
    print(' word2vec.syn0.shape={}'.format(word2vec.vectors.shape))
    print(' word2vec.syn0.mean={}'.format(emb_mean))
    print(' word2vec.syn0.std={}'.format(emb_std))

    nb_words = len(word_index)
    print('nb_words={}'.format(nb_words))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i > nb_words:
            continue
        if word in word2vec.vocab:
            embedding_vector = word2vec.word_vec(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                count += 1
            else:
                embedding_vector = word2vec.word_vec(word.capitalize())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    count += 1
                else:
                    embedding_vector = word2vec.word_vec(word.upper())
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        count += 1
                    else:
                        embedding_vector = word2vec.word_vec(word.lower())
                        if embedding_vector is not None:
                            embedding_matrix[i] = embedding_vector
                            count += 1
    print('find embedding={}/{}'.format(count, nb_words))

    return embedding_matrix


def get_embedding_matrix(embedding_file, word_index):
    if args.embedding_file.endswith('.bin'):
        return get_embedding_matrix_from_binary_file(embedding_file, word_index)
    else:
        return get_embedding_matrix_from_txt_file(embedding_file, word_index)


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
    # parser.add_argument("--ngrams_word", type=int, nargs='*')
    # parser.add_argument("--min_freq_word", type=int, nargs='*')
    parser.add_argument("--max_size_word", type=int, nargs='*')

    # test
    parser.add_argument("--ngrams_word", default=[1], type=int, nargs='*')
    parser.add_argument("--min_freq_word", default=[2], type=int, nargs='*')

    parser.add_argument("--ngrams_char", type=int, nargs='*')
    parser.add_argument("--min_freq_char", type=int, nargs='*')
    parser.add_argument("--max_size_char", type=int, nargs='*')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument("--num_char_no_split", action='store_true')
    parser.add_argument('--max_sent_len_ratio', default=0.99971, type=float)
    parser.add_argument('--max_sent_len', type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--embedding_file', default=r'D:\workspace_of_python\pretrained_language_model\Chinese-Word-Vectors\merge_sgns_bigram_char300.txt', type=str)
    args, _ = parser.parse_known_args()
    print(args)
    # exit(0)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device={}'.format(device))

    start_time = time.time()
    print("loading data...")
    data_train = './data/tnews_public/train.json'
    data_test = './data/tnews_public/dev.json'  # test.json has no label, using dev.json repalced.
    sentences_word, sentences_char, labels = build_dataset(data_train, num_char_no_split=args.num_char_no_split)
    data_train_word, data_dev_word, data_train_char, data_dev_char, labels_train, labels_dev = train_test_split(
        sentences_word, sentences_char, labels, test_size=0.1, shuffle=True, stratify=labels
    )
    # shuffle train/dev
    data_train_word, data_train_char, labels_train = shuffle_data(data_train_word, data_train_char, labels_train)
    data_dev_word, data_dev_char, labels_dev = shuffle_data(data_dev_word, data_dev_char, labels_dev)
    data_test_word, data_test_char, labels_test = build_dataset(data_test, num_char_no_split=args.num_char_no_split)

    print('train size={}'.format(len(data_train_word)))
    print('dev size={}'.format(len(data_dev_word)))
    print('test size={}'.format(len(data_test_word)))

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

    # MAX_VOCAB_SIZE = 10000
    UNK, PAD = '<UNK>', '<PAD>'

    x_train_index = None
    x_dev_index = None
    x_test_index = None
    vocabs_word = []
    vocabs_char = []
    ngrams_word = None
    if args.ngrams_word:
        ngrams_word = args.ngrams_word
        # if not isinstance(ngrams_word, list): ngrams_word=[ngrams_word]
        ngrams_word.sort()
    ngrams_char = None
    if args.ngrams_char:
        ngrams_char = args.ngrams_char
        # if not isinstance(ngrams_char, list): ngrams_char = [ngrams_char]
        ngrams_char.sort()

    if args.ngrams_word and args.ngrams_char:
        x_train_word_index_ngram, vocabs_word = build_x_index(data_train_word, ngrams_word, vocabs=vocabs_word,
                                                              max_size=args.max_size_word, min_freq=args.min_freq_word)
        x_dev_word_index_ngram, _ = build_x_index(data_dev_word, ngrams_word, vocabs=vocabs_word,
                                                  max_size=args.max_size_word, min_freq=args.min_freq_word)
        x_test_word_index_ngram, _ = build_x_index(data_test_word, ngrams_word, vocabs=vocabs_word,
                                                   max_size=args.max_size_word, min_freq=args.min_freq_word)

        append_idx = 0
        for v in vocabs_word:
            append_idx += len(v)

        x_train_char_index_ngram, vocabs_char = build_x_index(data_train_char, ngrams_char, vocabs=vocabs_char,
                                                              idx_start=append_idx,
                                                              max_size=args.max_size_char, min_freq=args.min_freq_char)
        x_dev_char_index_ngram, _ = build_x_index(data_dev_char, ngrams_char, vocabs=vocabs_char,
                                                  max_size=args.max_size_char, min_freq=args.min_freq_char)
        x_test_char_index_ngram, _ = build_x_index(data_test_char, ngrams_char, vocabs=vocabs_char,
                                                   max_size=args.max_size_char, min_freq=args.min_freq_char)

        x_train_index = row_concate(x_train_word_index_ngram, x_train_char_index_ngram)
        x_dev_index = row_concate(x_dev_word_index_ngram, x_dev_char_index_ngram)
        x_test_index = row_concate(x_test_word_index_ngram, x_test_char_index_ngram)
    elif args.ngrams_word:
        x_train_index, vocabs_word = build_x_index(data_train_word, ngrams_word, vocabs=vocabs_word,
                                                   max_size=args.max_size_word, min_freq=args.min_freq_word)
        x_dev_index, _ = build_x_index(data_dev_word, ngrams_word, vocabs=vocabs_word,
                                       max_size=args.max_size_word, min_freq=args.min_freq_word)
        x_test_index, _ = build_x_index(data_test_word, ngrams_word, vocabs=vocabs_word,
                                        max_size=args.max_size_word, min_freq=args.min_freq_word)
    elif args.ngrams_char:
        x_train_index, vocabs_char = build_x_index(data_train_char, ngrams_char, vocabs=vocabs_char, max_size=args.max_size_char, min_freq=args.min_freq_char)
        x_dev_index, _ = build_x_index(data_dev_char, ngrams_char, vocabs=vocabs_char, max_size=args.max_size_char, min_freq=args.min_freq_char)
        x_test_index, _ = build_x_index(data_test_char, ngrams_char, vocabs=vocabs_char, max_size=args.max_size_char, min_freq=args.min_freq_char)
    else:
        print('error, ngrams_word or ngrams_char necessary!')

    max_sent_len = 0  # default max len
    for i in x_train_index:
        if len(i) > max_sent_len:
            max_sent_len = len(i)
    print('max_sent_len={}'.format(max_sent_len))
    if args.max_sent_len_ratio and 0 < args.max_sent_len_ratio < 1:
        max_sent_len = cal_sent_len(x_train_index, args.max_sent_len_ratio)
        print('max_sent_len={}'.format(max_sent_len))
    if args.max_sent_len:
        max_sent_len = args.max_sent_len
        print('max_sent_len={}'.format(max_sent_len))
    # exit(0)

    pad_index = 0
    if args.ngrams_word:
        for idx, v in enumerate(vocabs_word):
            print('vocabs_word_{} size={}'.format(idx, len(v)))
            pad_index += len(v)
    if args.ngrams_char:
        for idx, v in enumerate(vocabs_char):
            print('vocabs_char_{} size={}'.format(idx, len(v)))
            pad_index += len(v)
    print('pad_index={}'.format(pad_index))

    x_train_index, x_train_seq_lens = sent_pad(x_train_index, max_sent_len, pad_index)
    x_dev_index, x_dev_seq_lens = sent_pad(x_dev_index, max_sent_len, pad_index)
    x_test_index, x_test_seq_lens = sent_pad(x_test_index, max_sent_len, pad_index)

    embedding_matrix = None
    if args.ngrams_word and not args.ngrams_char and args.ngrams_word == [1] and args.embedding_file:
        word_index = {}
        for n_gram_word, i in vocabs_word[0].items():
            word_index[n_gram_word[0]] = i
        word_index[PAD] = len(word_index)
        embedding_matrix = get_embedding_matrix(args.embedding_file, word_index)
        embedding_matrix = torch.tensor(embedding_matrix)

    # # test
    # x_train_index = x_train_index[:32+24]
    # x_dev_index = x_dev_index[:32+24]
    # x_test_index = x_test_index[:32+24]
    # x_train_seq_lens = x_train_seq_lens[:32 + 24]
    # x_dev_seq_lens = x_dev_seq_lens[:32 + 24]
    # x_test_seq_lens = x_test_seq_lens[:32 + 24]
    # y_train_index = y_train_index[:32 + 24]
    # y_dev_index = y_dev_index[:32 + 24]
    # y_test_index =y_test_index[:32 + 24]

    # model = FastTextModel(pad_index + 1, 300, 0.2, num_classes)
    # model = TextCNN1DModel(pad_index + 1, 300, 256, (2, 3, 4), 0.2, num_classes)
    # model = TextCNN2DModel(pad_index + 1, 300, 256, (2, 3, 4), 0.2, num_classes)
    model = TextLSTMModel(pad_index + 1, 300, 256, 2, 0.2, num_classes)
    print(model)
    model = model.to(device)
    learning_rate = args.learning_rate
    best_acc = 0
    start_epoch = 0

    # resume = False
    # if resume:
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     model.load_state_dict(checkpoint['model'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # trainset
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batch = 0
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        for batch_idx, (inputs, targets) in enumerate(
                DatasetIterater((x_train_index, x_train_seq_lens), y_train_index, batch_size=args.batch_size, device=device)
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

        # devset
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        dev_batch = 0
        # for batch_idx, (inputs, targets) in enumerate(devloader):
        for batch_idx, (inputs, targets) in enumerate(
                DatasetIterater((x_dev_index, x_dev_seq_lens), y_dev_index, batch_size=args.batch_size, device=device)
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

        print('epoch: {}/{}, train loss={:.4f}, train acc={:.2f}%, dev loss={:.4f}, dev acc={:.2f}%'.format(
            epoch + 1, start_epoch + args.epochs,
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

            # testset
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_batch = 0
            # for batch_idx, (inputs, targets) in enumerate(testloader):
            for batch_idx, (inputs, targets) in enumerate(
                    DatasetIterater((x_test_index, x_test_seq_lens), y_test_index, batch_size=args.batch_size, device=device)
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

        # scheduler.step()

    end_time = time.time()
    print('time used={:.1f}s'.format(end_time - start_time))
