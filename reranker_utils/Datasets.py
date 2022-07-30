import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from reranker_utils.utils import truncate_sequences


class PretrainDataset(Dataset):

    def __init__(self, corpus, seq_len: int, tokenizer):
        self.seq_len = seq_len - 3
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = tokenizer
        self.vocab = {index: token for token, index in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2 = self.get_sentence(idx)
        text_pair = ['[CLS]'] + text1 + ['[SEP]'] + text2 + ['[SEP]']
        text_pair, output_ids = self.create_masked_lm_predictions(text_pair)
        token_ids = []
        for token in text_pair:
            if token == '[CLS]' or '[SEP]':
                token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            else:
                temp_char = self.tokenizer.tokenize(token.lower())
                if temp_char:
                    token_ids.append(self.tokenizer.convert_tokens_to_ids(temp_char[0]))
                else:
                    token_ids.append(self.tokenizer.convert_tokens_to_ids("§"))
        segment_ids = [0] * len(token_ids)

        padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
        padding_label = [-100 for _ in range(self.seq_len + 3 - len(token_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        token_ids.extend(padding), output_ids.extend(padding_label), segment_ids.extend(padding)
        attention_mask = np.array(attention_mask)
        token_ids = np.array(token_ids)
        segment_ids = np.array(segment_ids)
        output_ids = np.array(output_ids)
        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  'attention_mask': attention_mask,
                  "output_ids": output_ids}
        return output

    def get_sentence(self, idx):

        text1, text2, _ = self.lines[idx]
        text1, text2 = truncate_sequences(self.seq_len, -1, text1, text2)

        return text1, text2

    def create_masked_lm_predictions(self, text, masked_lm_prob=0.15, max_predictions_per_seq=512,
                                     rng=random.Random()):
        cand_indexes = []
        for (i, token) in enumerate(text):
            if token == '[CLS]' or token == '[SEP]':
                continue
            cand_indexes.append([i])

        output_tokens = text
        output_tokens_copy = output_tokens.copy()
        output_labels = [-100] * len(text)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(text) * masked_lm_prob))))

        # 不同 gram 的比例  **(改为3)**
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        # 每个 token 对应的三个 ngram
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        rng.shuffle(ngram_indexes)

        masked_lms = set()
        # 获取 masked tokens
        # cand_index_set 其实就是每个 token 的三个 ngram
        # 比如：[[[13]], [[13], [14]], [[13], [14], [15]]]
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # 根据 cand_index_set 不同长度 choice
            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            # [16, 17] = sum([[16], [17]], [])
            index_set = sum(cand_index_set[n - 1], [])
            # 处理选定的 ngram index ：80% MASK，10% 是原来的，10% 随机替换一个
            for index in index_set:
                masked_token = None
                if rng.random() < 0.8:
                    masked_token = '[MASK]'
                else:
                    if rng.random() < 0.5:
                        masked_token = text[index]
                    else:
                        masked_token = self.vocab[
                            rng.randint(0, self.tokenizer.vocab_size - (106 + 1))]  # 取不到特殊字符
                        output_labels[index] = self.vocab.get(output_tokens[index], 1)
                temp_char = self.tokenizer.tokenize(output_tokens[index].lower())
                if temp_char:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids(temp_char[0])
                else:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids("§")
                output_tokens_copy[index] = masked_token
                masked_lms.add(index)

        return output_tokens_copy, output_labels


class FineTuneDataset(Dataset):

    def __init__(self, corpus, args, tokenizer):
        self.q_maxlen = args.q_maxlen
        self.p_head_maxlen = args.p_head_maxlen
        self.p_tail_maxlen = args.p_tail_maxlen
        self.p_maxlen = self.p_head_maxlen + self.p_tail_maxlen
        self.seq_len = self.q_maxlen + self.p_maxlen
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2, label = self.get_sentence_and_label(idx)
        text1_token_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        text2_token_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        text1_token_ids, text2_token_ids = truncate_sequences(self.q_maxlen, self.p_head_maxlen, self.p_tail_maxlen,
                                                              text1_token_ids,
                                                              text2_token_ids)
        token_ids = [101] + text1_token_ids + [102] + text2_token_ids + [102]
        segment_ids = [0] * (len(text1_token_ids) + 2) + [1] * (len(text2_token_ids) + 1)

        padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        token_ids.extend(padding), segment_ids.extend(padding)
        # attention_mask = np.array(attention_mask)
        # token_ids = np.array(token_ids)
        # segment_ids = np.array(segment_ids)
        # label = np.array(label)
        output = {"input_ids": torch.tensor(token_ids),
                  "token_type_ids": torch.tensor(segment_ids),
                  'attention_mask': torch.tensor(attention_mask),
                  "label": torch.tensor(label)}
        return output

    def get_sentence_and_label(self, idx):
        text1, text2, label = self.lines[idx]

        return text1, text2, int(label)


class FineTuneDatasetForListWise(Dataset):

    def __init__(self, corpus, args, tokenizer):
        self.q_maxlen = args.q_maxlen
        self.p_head_maxlen = args.p_head_maxlen
        self.p_tail_maxlen = args.p_tail_maxlen
        self.p_maxlen = self.p_head_maxlen + self.p_tail_maxlen
        self.seq_len = self.q_maxlen + self.p_maxlen
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        examples = self.lines[idx]
        encoded_examples = []
        for item_id, item in enumerate(examples):
            label = 1 if item_id == 0 else 0
            query, passage = item[0], item[1]
            query_token_ids = self.tokenizer.encode(query, add_special_tokens=False)
            passage_token_ids = self.tokenizer.encode(passage, add_special_tokens=False)
            query_token_ids, passage_token_ids = truncate_sequences(self.q_maxlen, self.p_head_maxlen,
                                                                    self.p_tail_maxlen, query_token_ids,
                                                                    passage_token_ids)
            token_ids = [101] + query_token_ids + [102] + passage_token_ids + [102]
            segment_ids = [0] * (len(query_token_ids) + 2) + [1] * (len(passage_token_ids) + 1)

            padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
            attention_mask = len(token_ids) * [1] + len(padding) * [0]
            token_ids.extend(padding), segment_ids.extend(padding)
            encoded_example = [token_ids, segment_ids, attention_mask, label]
            encoded_examples.append(encoded_example)
        return encoded_examples

    @classmethod
    def collate(cls, batch):
        token_ids, segment_ids, attention_mask, label = [], [], [], []
        for item in batch:
            for i in item:
                token_ids.append(i[0])
                segment_ids.append(i[1])
                attention_mask.append(i[2])
                label.append(i[3])

        return {"input_ids": torch.tensor(token_ids),
                "token_type_ids": torch.tensor(segment_ids),
                'attention_mask': torch.tensor(attention_mask),
                "label": torch.tensor(label)}


class FineTuneDatasetForDynamicLen(Dataset):
    """
    动态句长
    """

    def __init__(self, corpus, vocab: dict, args):
        self.vocab = vocab
        self.corpus = corpus
        self.args = args
        self.examples = self.processor()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.examples[0][idx], self.examples[1][idx]

    @staticmethod
    def collate_fn(data):
        data.sort(key=lambda x: - len(x[0]))
        data = list(zip(*data))
        id_batch = data[0]
        label_batch = np.array(data[1])
        return [id_batch, label_batch]

    def processor(self):
        id_list, label_list = [], []
        for text, label in self.corpus:
            text_ids = [self.vocab.get(t, 1) for t in text]
            token_ids = ([2] + text_ids + [3])
            id_list.append(torch.tensor([token_ids]).to(self.args.device).long())
            label_list.append(label)

        return [id_list, label_list]


class PretrainDatasetForBlockShuffle(Dataset):
    """
    分块shuffle
    """

    def __init__(self, corpus, seq_len: int, tokenizer):
        self.seq_len = seq_len - 3
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = tokenizer
        self.vocab = {index: token for token, index in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2 = self.get_sentence(idx)
        text_pair = ['[CLS]'] + text1 + ['[SEP]'] + text2 + ['[SEP]']
        text_pair, output_ids = self.create_masked_lm_predictions(text_pair)
        token_ids = []
        for token in text_pair:
            if token == '[CLS]' or '[SEP]':
                token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            else:
                temp_char = self.tokenizer.tokenize(token.lower())
                if temp_char:
                    token_ids.append(self.tokenizer.convert_tokens_to_ids(temp_char[0]))
                else:
                    token_ids.append(self.tokenizer.convert_tokens_to_ids("§"))
        segment_ids = [0] * (len(text1) + 2) + [1] * (len(text2) + 1)

        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  "output_ids": output_ids}
        return output

    def get_sentence(self, idx):

        text1, text2, _ = self.lines[idx]
        text1, text2 = list(text1), list(text2)
        text1, text2 = truncate_sequences(self.seq_len, -1, text1, text2)

        return text1, text2

    def create_masked_lm_predictions(self, text, masked_lm_prob=0.15, max_predictions_per_seq=512,
                                     rng=random.Random()):
        cand_indexes = []
        for (i, token) in enumerate(text):
            if token == '[CLS]' or token == '[SEP]':
                continue
            cand_indexes.append([i])

        output_tokens = text
        output_tokens_copy = output_tokens.copy()
        output_labels = [-100] * len(text)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(text) * masked_lm_prob))))

        # 不同 gram 的比例  **(改为3)**
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        # 每个 token 对应的三个 ngram
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        rng.shuffle(ngram_indexes)

        masked_lms = set()
        # 获取 masked tokens
        # cand_index_set 其实就是每个 token 的三个 ngram
        # 比如：[[[13]], [[13], [14]], [[13], [14], [15]]]
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # 根据 cand_index_set 不同长度 choice
            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            # [16, 17] = sum([[16], [17]], [])
            index_set = sum(cand_index_set[n - 1], [])
            # 处理选定的 ngram index ：80% MASK，10% 是原来的，10% 随机替换一个
            for index in index_set:
                masked_token = None
                if rng.random() < 0.8:
                    masked_token = '[MASK]'
                else:
                    if rng.random() < 0.5:
                        masked_token = text[index]
                    else:
                        masked_token = self.vocab[
                            rng.randint(0, self.tokenizer.vocab_size - (106 + 1))]  # 取不到特殊字符
                        output_labels[index] = self.vocab.get(output_tokens[index], 1)
                temp_char = self.tokenizer.tokenize(output_tokens[index].lower())
                if temp_char:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids(temp_char[0])
                else:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids("§")
                output_tokens_copy[index] = masked_token
                masked_lms.add(index)

        return output_tokens_copy, output_labels

    @classmethod
    def padding_list(cls, ls, val, returnTensor=False):
        ls = ls[:]
        max_len = max([len(i) for i in ls])
        for i in range(len(ls)):
            ls[i] = ls[i] + [val] * (max_len - len(ls[i]))
        return torch.tensor(ls) if returnTensor else ls

    @classmethod
    def collate(cls, batch):
        input_ids = [i['input_ids'] for i in batch]
        token_type_ids = [i['token_type_ids'] for i in batch]
        output_ids = [i['output_ids'] for i in batch]
        input_ids = PretrainDatasetForBlockShuffle.padding_list(input_ids, 0, returnTensor=True)
        token_type_ids = PretrainDatasetForBlockShuffle.padding_list(token_type_ids, 0, returnTensor=True)
        output_ids = PretrainDatasetForBlockShuffle.padding_list(output_ids, -100, returnTensor=True)
        attention_mask = (input_ids != 0)
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'output_ids': output_ids}


if __name__ == '__main__':
    import json
    from tqdm import tqdm
    from transformers import BertTokenizer


    def truncate_sequences(maxlen, index, *sequences):
        """
        截断总长度至不超过maxlen
        """
        sequences = [s for s in sequences if s]
        while True:
            lengths = [len(s) for s in sequences]
            if sum(lengths) > maxlen:
                i = np.argmax(lengths)
                sequences[i].pop(index)
            else:
                return sequences


    tokenizer = BertTokenizer(
        vocab_file='/Notebook/data_801/chenhy/MODEL/bert_wwm/vocab.txt',
        do_lower_case=True)
    filename = '../data/reranker_train.json'
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    for item in tqdm(data):
        examples = []
        query = item['qry']['query']
        pos_psg = random.choice([i['passage'] for i in item['pos']])
        examples.append((query, pos_psg))

        if len(item['neg']) < 8 - 1:
            negs = random.choices([i['passage'] for i in item['neg']], k=8 - 1)
        else:
            negs = random.sample([i['passage'] for i in item['neg']], k=8 - 1)

        for neg_psg in negs:
            examples.append((query, neg_psg))

        D.append(examples)

    for item in tqdm(D):
        query, passage = item[0], item[1]
        query_token_ids = tokenizer.encode(query, add_special_tokens=False)
        passage_token_ids = tokenizer.encode(passage, add_special_tokens=False)
        query_token_ids, passage_token_ids = truncate_sequences(384, -1, query_token_ids,
                                                                passage_token_ids)
        if len(passage_token_ids) < 1 or len(query_token_ids) < 1:
            print(query, passage, query_token_ids, passage_token_ids)
