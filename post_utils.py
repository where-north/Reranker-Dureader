"""
Name : post_utils.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/23 8:19
Desc:
"""
import json
from collections import defaultdict
from tqdm import tqdm
import os
import ijson


def split_data(file_path):
    """
    把数据切成4份
    """
    main_path, file_name = os.path.split(file_path)
    out_file = os.path.splitext(file_name)[0]
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 数据格式：{'q_text': '问题', 'q_id': '问题ID', 'top_n': [['段落ID', '段落'],...]}
        objects = ijson.items(f, 'item')
        for item in tqdm(objects, desc=f'加载{file_path}'):
            data.append(item)
    print(f'data len: {len(data)}')
    sub_size = int(len(data) / 4)
    for i in range(4):
        start_id = i * sub_size
        end_id = len(data) if i == 3 else start_id + sub_size
        sub_data = data[start_id:end_id]
        print(f'保存索引为 {start_id}:{end_id} 的数据到 {main_path}/{out_file}_{i}')
        with open(f'{main_path}/{out_file}_{i}', 'w', encoding='utf-8') as w:
            json.dump(sub_data, w, ensure_ascii=False, indent=4)


def rerank_test(model_path=''):
    main_path = f'./reranker_model/{model_path}/'
    top_n_data = defaultdict(dict)
    for i in range(4):
        with open(main_path + f'reranker_scores_{i}.json', 'r') as f:
            sub_data = json.load(f)
            top_n_data.update(sub_data)

    """加载top50数据
    单条格式：
    'q_id':{'q_text': '',
   'top_n': [(doc_id, doc_text, doc_score), (...)]}
    """
    sort_res = defaultdict(list)
    for q_id in top_n_data.keys():
        top_n = top_n_data[q_id]['top_n']
        sort_top_n = sorted(top_n, key=lambda x: x[2], reverse=True)
        sort_res[q_id] = [i[0] for i in sort_top_n]

    with open('./sort_test_res.json', 'w', encoding='utf-8') as f:
        json.dump(sort_res, f)


def rerank_valid(model_path=''):
    main_path = f'./reranker_model/{model_path}/'

    scores = json.load(open(main_path + f'val_scores_.json', 'r'))

    """加载top50数据
    单条格式：
    {'q_text': '',
   'q_id': '',
   'top_n': [(doc_id, doc_text), (...)]}
    """
    with open('./data/dev_data_top50.json', 'r') as f:
        top50_data = json.load(f)
    sort_res = defaultdict(list)
    for line, score in zip(top50_data, scores):
        q_id = line['q_id']
        top_n = line['top_n']
        # 对所有段落进行排序
        for i in range(len(top_n)):
            top_n[i][1] = score[i][0]
        sort_top_n = sorted(top_n, key=lambda x: x[1], reverse=True)
        sort_top_n_id = [i[0] for i in sort_top_n]
        sort_res[q_id] = sort_top_n_id

    search_dev_res_data = {}
    for line in top50_data:
        top_n_ids = [i[0] for i in line['top_n']]
        search_dev_res_data[line['q_id']] = top_n_ids

    calculate_dev_mrr(sort_res, search_dev_res_data)


def calculate_dev_mrr(sort_res, search_dev_res_data):
    dev_data_path = './data/dev.json'
    dev_data_file = open(dev_data_path, 'r', encoding='utf-8')
    dev_data = [json.loads(i) for i in dev_data_file.readlines()]

    assert len(search_dev_res_data) == len(sort_res)

    search_mrr = 0
    rerank_mrr = 0

    for item in tqdm(dev_data):
        qid = item['question_id']
        pos_paragraph_ids = [i['paragraph_id'] for i in item['answer_paragraphs']]
        for idx, paragraph_id in enumerate(search_dev_res_data[qid]):
            if paragraph_id in pos_paragraph_ids:
                search_mrr += (1 / (idx + 1))
                break
        for idx, paragraph_id in enumerate(sort_res[qid]):
            if paragraph_id in pos_paragraph_ids:
                rerank_mrr += (1 / (idx + 1))
                break

    print(f'search mrr: {search_mrr / len(search_dev_res_data)}')
    print(f'rerank mrr: {rerank_mrr / len(sort_res)}')


def rerank_train(model_path=''):
    main_path = f'./reranker_model/{model_path}/'
    # 内存有限，分4波处理
    for i in range(4):
        top_n_data = defaultdict(dict)
        print(f'加载 train_scores_{i}.json')
        with open(main_path + f'train_scores_{i}.json', 'r') as f:
            sub_data = json.load(f)
            top_n_data.update(sub_data)

        for q_id in top_n_data.keys():
            top_n = top_n_data[q_id]['top_n']

            # 对所有段落进行排序
            sort_top_n = sorted(top_n, key=lambda x: x[2], reverse=True)
            top_n_data[q_id]['top_n'] = sort_top_n

        with open(f'./reranker_model/{model_path}/sorted_train_scores_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(top_n_data, f)


if __name__ == '__main__':
    split_data(file_path='./data/test2_0.693/test2_data_top50.json')
    # rerank_train('macbert_large_pointwise_0.740')
    # rerank_valid(model_path='2022-05-01_04_14_55')
    # rerank_test(model_path='2022-06-23_15_23_14')


