# -*- coding: utf-8 -*-
"""
Name : utils.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2021/8/4 20:32
Desc:
"""
import os
import pandas as pd
import logging
import datetime
import random
import torch
import numpy as np
import json

# def replace_char(char):
#     char = char.replace(',', '，')
#     char = char.replace('?', '？')
#     char = char.replace('!', '！')
#
#     return char

random.seed(0)


def load_pontwise_data(filename):
    """加载数据
    单条格式：`query null para_text label` (`\t` seperated, `null` represents invalid column.)
    """
    data = pd.read_csv(filename, sep='\t', names=['question', 'nan', 'passage', 'label'])
    data = data.drop('nan', axis=1)
    data = data.dropna()
    D = []
    if 'label' in data.columns:
        for text1, text2, label in zip(data['question'], data['passage'], data['label']):
            D.append((text1, text2, int(label)))
    else:
        for text1, text2 in zip(data['question'], data['passage']):
            label = -100
            D.append((text1, text2, label))
    return D


def load_listwise_train_data(args):
    """加载数据
    """
    filename = args.train_data_path
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    for item in data:
        examples = []
        query = item['qry']['query']
        pos_psg = random.choice([i['passage'] for i in item['pos']])
        examples.append((query, pos_psg))

        if len(item['neg']) < args.train_group_size - 1:
            negs = random.choices([i['passage'] for i in item['neg']], k=args.train_group_size - 1)
        else:
            negs = random.sample([i['passage'] for i in item['neg']], k=args.train_group_size - 1)

        for neg_psg in negs:
            examples.append((query, neg_psg))

        D.append(examples)
    return D


def load_listwise_valid_data(args):
    """加载数据
    """
    filename = args.valid_data_path
    p_ids, pos_ids, D = [], [], []
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    for item in data:
        query = item['q_text']
        pos_ids.append(item['pos_ids'])
        count = 0
        temp = []
        for i in item['top_n']:
            if count >= args.valid_n_docs:
                break
            pid, text = i[0], i[1]
            temp.append(pid)
            if len(text) == 0:
                print(f'skip empty doc_text: {pid}')
                continue
            if pid in item['pos_ids']:
                D.append((query, text, 1))
            else:
                D.append((query, text, 0))
            count += 1
        p_ids.append(temp)

    return p_ids, pos_ids, D


def load_test_data(filename, n_docs=50):
    """加载测试数据
    单条格式：
    {'q_text': '',
   'q_id': '',
   'top_n': [(doc_id, doc_text), (...)]}
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    D, label = [], -100
    q_ids, doc_ids = [], []
    q_texts, doc_texts = [], []
    for line in data:
        q_id = line['q_id']
        q_text = line['q_text']
        doc_id_texts = line['top_n']
        count = 0
        for item in doc_id_texts:
            if count >= n_docs:
                break
            doc_id, doc_text = item[0], item[1]
            if len(doc_text) == 0:
                print(f'skip empty doc_text: {doc_id}')
                continue
            D.append((q_text, doc_text, label))
            q_ids.append(q_id)
            doc_ids.append(doc_id)
            q_texts.append(q_text)
            doc_texts.append(doc_text)
            count += 1

    return q_ids, doc_ids, q_texts, doc_texts, D


def get_save_path(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    args.model_save_path = args.model_save_path + "{}/".format(timestamp)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    argsDict = args.__dict__
    with open(args.model_save_path + 'args.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def get_logger(log_file, display2console=True):
    """
    定义日志方法
    :param display2console:
    :param log_file:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为DEBUG
    logger.setLevel(logging.DEBUG)
    # 创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 创建一个控制台的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    if display2console:
        logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def truncate_sequences(maxlen, index, *sequences):
#     """
#     截断总长度至不超过maxlen
#     """
#     sequences = [s for s in sequences if s]
#     while True:
#         lengths = [len(s) for s in sequences]
#         if sum(lengths) > maxlen:
#             i = np.argmax(lengths)
#             sequences[i].pop(index)
#         else:
#             return sequences

def truncate_sequences(q_maxlen, p_head_maxlen, p_tail_maxlen, *sequences):
    """
    截断总长度至不超过maxlen
    """
    q_tokens, p_tokens = sequences[0], sequences[1]
    q_len, p_len = len(q_tokens), len(p_tokens)
    if q_len > q_maxlen:
        q_tokens = q_tokens[:q_maxlen]
    if p_len > p_head_maxlen + p_tail_maxlen:
        p_tokens = p_tokens[:p_head_maxlen] + p_tokens[p_len-p_tail_maxlen:]
    if p_len == 0:
        print('警告：出现空段落！')
        p_tokens = [100]

    return q_tokens, p_tokens


class ModelSaver:
    def __init__(self, args, patience=10000, metric='max'):
        """
        :param args:
        :param patience: 连续几个epoch无提升，停止训练
        """
        if metric == 'max':
            self.best_score = float('-inf')
        elif metric == 'min':
            self.best_score = float('inf')
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        self.args = args
        self.metric = metric
        self.bad_perform_count = 0
        self.patience = patience

    def save(self, eval_score, epoch, step, logger, model):
        do_save = None
        if self.metric == 'max':
            do_save = eval_score > self.best_score
        elif self.metric == 'min':
            do_save = eval_score < self.best_score
        if do_save:
            self.bad_perform_count = 0
            self.best_score = eval_score
            save_path = self.args.model_save_path + f'/{self.args.model_type}_best_model.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
        else:
            self.bad_perform_count += 1
            if self.bad_perform_count > self.patience:
                return True
        save_path = self.args.model_save_path + f'/{self.args.model_type}_epoch{epoch + 1}_step{step}.pth'
        torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
        logger.info(f'save model in {save_path}')
