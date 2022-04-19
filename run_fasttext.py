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
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn
import codecs


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
            sentences_word.append(jieba.cut(sentence))
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


def build_vocab(sentences, max_size=None, min_freq=2):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            vocab[word] = vocab.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
    if max_size and max_size<len(vocab_list):
        vocab_list = vocab_list[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def create_ngram(sent, ngram_value):
    return set(zip(*[sent[i:] for i in range(ngram_value)]))


def build_x_index(sentences, vocab, ngrams):
    x_index_1_gram = [] # 1-gram necessary
    for sentence in sentences:
        line_index = []
        for word in sentence:
            line_index.append(vocab.get(word))
        x_index_1_gram.append(x_index_1_gram)
    for n in ngrams:
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--word_or_char", default="word", type=str)     # word,char,word_and_char
    parser.add_argument("--word_ngram", type=list, nargs='*')
    parser.add_argument("--char_ngram", type=list, nargs='*')
    parser.add_argument("--monitor", default="val_loss", type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument("--embedding_file", default='', type=str)
    parser.add_argument('--number_words', default=sys.maxsize, type=int)
    parser.add_argument('--number_chars', default=sys.maxsize, type=int)
    parser.add_argument("--trainable", action='store_true')
    parser.add_argument("--class_weights", action='store_true')
    parser.add_argument("--num_char_no_split", action='store_true')
    parser.add_argument("--max_word_length", default=sys.maxsize, type=int)
    args, _ = parser.parse_known_args()
    print(args)
    exit(0)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    start_time = time.time()
    print("loading data...")
    data_train = './data/tnews_public/train.json'
    data_test = './data/tnews_public/dev.json'  # test.json has no label, using dev.json repalced.
    sentences_word, sentences_char, labels = build_dataset(data_train, num_char_no_split=args.num_char_no_split)
    data_train_word, data_dev_word, data_train_char, data_dev_char, labels_train, labels_dev = train_test_split(
        sentences_word, sentences_char, labels, test_size=0.1, shuffle=True, stratify=labels
    )
    data_test_word, data_test_char, labels_test = build_dataset(data_test, num_char_no_split=args.num_char_no_split)

    MAX_VOCAB_SIZE = 10000
    UNK, PAD = '<UNK>', '<PAD>'
    vocab_word_idx_dict, vocab_char_idx_dict = None, None
    if 'word' in args.word_or_char:
        vocab_word_idx_dict = build_vocab(data_train_word, max_size=args.number_words)
    if 'char' in args.word_or_char:
        vocab_char_idx_dict = build_vocab(data_train_char, max_size=args.number_chars)

    if args.word_ngram:
        ngrams = args.word_ngram
        ngrams.sort()
        x_train_word_index_ngram = build_x_index(data_train_word, vocab_word_idx_dict, ngrams)
        x_dev_word_index_ngram = build_x_index(data_dev_word, vocab_word_idx_dict, ngrams)
        x_test_word_index_ngram = build_x_index(data_test_word, vocab_word_idx_dict, ngrams)


    end_time = time.time()
    print('time used={:.1f}s'.format(end_time - start_time))
