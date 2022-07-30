"""
Name : prepare_train_data.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/26 21:56
Desc:
"""
import json
import random
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import ijson
import os

'''
单条训练数据格式：`query null para_text label` (`\t` seperated, `null` represents invalid column.)
为每条问题随机采样4条负样本。
'''


def get_pointwise_train_data(retrieve_train_data_path, origin_train_data_path='./train.json'):
    """
    @param retrieve_train_data_path: 检索模型对训练集进行检索的结果
    @param origin_train_data_path: 原始训练集
    """
    random.seed(1234)
    print(f'加载 {retrieve_train_data_path}')
    top_n_data = []
    with open(retrieve_train_data_path, 'r', encoding='utf-8') as f:
        # 数据格式：{'q_text': '问题', 'q_id': '问题ID', 'top_n': [['段落ID', '段落'],...]}
        objects = ijson.items(f, 'item')
        for item in tqdm(objects, desc=f'加载{retrieve_train_data_path}'):
            top_n_data.append(item)

    print(f'加载 {origin_train_data_path}')
    with open(origin_train_data_path, 'r', encoding='utf-8') as f:
        # 数据格式：[{'question_id': '', 'question': '', 'answer_paragraphs': [{'paragraph_id': '', 'paragraph_text': ''}, ...]}]
        train_data = [json.loads(line) for line in f.readlines()]

    assert len(top_n_data) == len(train_data)

    query, para_text, label = [], [], []
    for i in tqdm(range(len(train_data))):
        if train_data[i]['question_id'] == top_n_data[i]['q_id']:
            question = train_data[i]['question']
            pos_para_pools = set([item['paragraph_text'] for item in train_data[i]['answer_paragraphs']])
            pos_para_pools = set(list(pos_para_pools))
            top_para_pools = set([item[1] for item in top_n_data[i]['top_n']])
            neg_para_pools = list(top_para_pools - pos_para_pools)
            # 随机选一条正例
            pos_para_pools = [random.choice(list(pos_para_pools))]
            sample_neg_number = min(len(pos_para_pools) * 4, len(neg_para_pools))
            # if sample_neg_number < len(pos_para_pools) * 4:
            #     print(train_data[i]['question'], len(top_para_pools), len(pos_para_pools), len(neg_para_pools))
            random.shuffle(neg_para_pools)
            para_text.extend([neg_para_pools[j] for j in range(sample_neg_number)])
            label.extend([0 for _ in range(sample_neg_number)])
            query.extend([question for _ in range(sample_neg_number)])
            for pos in pos_para_pools:
                query.append(question)
                para_text.append(pos)
                label.append(1)

    assert len(query) == len(para_text) == len(label)
    print(f'label 1 numbers: {sum(label)}, label 0 numbers: {len(label) - sum(label)}')

    sample_rerank_data = pd.DataFrame({
        'query': query,
        'null': ['' for _ in range(len(query))],
        'para_text': para_text,
        'label': label
    })

    sample_rerank_data = sample_rerank_data.sample(frac=1).reset_index(drop=True)
    data_len = len(sample_rerank_data)
    print(f'all data len: {data_len}')
    train_size = int(data_len * 0.9)
    train_data = sample_rerank_data.iloc[:train_size].reset_index(drop=True)
    valid_data = sample_rerank_data.iloc[train_size:].reset_index(drop=True)
    os.makedirs('./pointwise_train_data', exist_ok=True)
    pd.DataFrame(train_data).to_csv('./pointwise_train_data/reranker_train.tsv',
                                    sep='\t', index=False, header=None)
    pd.DataFrame(valid_data).to_csv('./pointwise_train_data/reranker_valid.tsv',
                                    sep='\t', index=False, header=None)


def get_listwise_train_data(retrieve_train_data_path, out_dir, sample_start=0, sample_end=200,
                            origin_train_data_path='./train.json', is_random=True, n_samples=7, seed=1234):
    """
    @param seed: 随机种子
    @param retrieve_train_data_path: 检索模型对训练集进行检索的结果
    @param out_dir: 训练集输出路径
    @param sample_start: 负样本采样范围的start id（[sample_start：sample_end]内随机采样负样本）
    @param sample_end: 负样本采样范围的end id
    @param origin_train_data_path: 原始训练集
    @param is_random: 是否随机采样
    @param n_samples: 为每个问题采样负样本的个数
    @return:
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    def read_qrel():
        nonlocal origin_train_data_path
        qrel = defaultdict(list)  # {qid:[pos_pids]}
        query_collection = defaultdict(str)  # {qid:q_text}
        doc_collection = defaultdict(str)  # {pid:p_text}
        with open(origin_train_data_path, 'r', encoding='utf-8') as f:
            # 数据格式：[{'question_id': '', 'question': '', 'answer_paragraphs': [{'paragraph_id': '', 'paragraph_text': ''}, ...]}]
            train_data = [json.loads(line) for line in f.readlines()]
        print(f'origin train query numbers: {len(train_data)}')
        for item in tqdm(train_data, desc=f'处理{origin_train_data_path}'):
            qid = item['question_id']
            query_collection[qid] = str(item['question'])
            for pos in item['answer_paragraphs']:
                pos_id = pos['paragraph_id']
                # text = str(pos['paragraph_text']).replace('百度经验:jingyan.baidu.com', '')
                text = str(pos['paragraph_text'])
                if len(text) < 1:
                    print(f'skip {pos_id} {str(pos["paragraph_text"])}')
                    continue
                doc_collection[pos_id] = text
                qrel[qid].append(pos_id)
        return qrel, query_collection, doc_collection

    qrel, query_collection, doc_collection = read_qrel()
    queries = list(qrel.keys())
    # 获取q的top n 召回段落（去除相关段落），以及将负样本段落添加到did对应的d_text对应字典。
    negs_pool = defaultdict(list)
    no_judge = set()
    top_n_data = []
    with open(retrieve_train_data_path, 'r', encoding='utf-8') as f:
        # 数据格式：{'q_text': '问题', 'q_id': '问题ID', 'top_n': [['段落ID', '段落'],...]}
        objects = ijson.items(f, 'item')
        for item in tqdm(objects, desc=f'加载{retrieve_train_data_path}'):
            top_n_data.append(item)
        for l in tqdm(top_n_data, desc=f'处理{retrieve_train_data_path}'):
            qid = l['q_id']
            for p in l['top_n']:
                p_id, p_text = p[0], p[1]
                if p_id in qrel[qid]:
                    continue
                if '￡是什么货币' in l["q_text"] and len(p_text) < 10:
                    continue
                # p_text = p_text.replace('百度经验:jingyan.baidu.com', '')
                if len(p_text) < 1:
                    print(f'skip {l["q_text"]} {p_id} {p[1]}')
                    continue
                negs_pool[qid].append(p_id)
                doc_collection[p_id] = p_text

    print(f'{len(no_judge)} queries not judged and skipped', flush=True)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + 'reranker_train.json', 'w') as f:
        for qid in tqdm(queries, desc=f'构造{out_dir + "reranker_train.json"}'):
            # pick from top of the full initial ranking
            negs = negs_pool[qid][sample_start:sample_end]
            # shuffle if random flag is on
            if is_random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:n_samples]

            neg_dict = []
            for neg in negs:
                doc_text = doc_collection[neg]
                neg_dict.append({
                    'pid': neg,
                    'passage': doc_text if doc_text else '',
                })
            pos_dict = []
            for pos in qrel[qid]:
                doc_text = doc_collection[pos]
                pos_dict.append({
                    'pid': pos,
                    'passage': doc_text if doc_text else '',
                })
            query_dict = {
                'qid': qid,
                'query': query_collection[qid],
            }
            item_set = {
                'qry': query_dict,
                'pos': pos_dict,
                'neg': neg_dict,
            }
            f.write(json.dumps(item_set, ensure_ascii=False) + '\n')


def get_listwise_valid_data(retrieve_dev_data_path, out_dir, origin_dev_data_path='./dev.json'):
    """
    @param retrieve_dev_data_path: 检索模型对验证集进行检索的结果
    @param out_dir: 验证集输出路径
    @param origin_dev_data_path: 原始验证集
    @return:
    """
    random.seed(1234)
    with open(origin_dev_data_path, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f.readlines()]
    each_qid_pid_dict = defaultdict(list)
    for item in tqdm(dev_data, desc=f'加载 {origin_dev_data_path}'):
        qid = item['question_id']
        pids = [i['paragraph_id'] for i in item['answer_paragraphs']]
        each_qid_pid_dict[qid] = pids

    top_n_data = []
    with open(retrieve_dev_data_path, 'r', encoding='utf-8') as f:
        # 数据格式：{'q_text': '问题', 'q_id': '问题ID', 'top_n': [['段落ID', '段落'],...]}
        objects = ijson.items(f, 'item')
        for item in tqdm(objects, desc=f'加载{retrieve_dev_data_path}'):
            top_n_data.append(item)
    with open(out_dir + 'reranker_valid.json', 'w') as f:
        for item in tqdm(top_n_data, desc=f'构造 {out_dir + "reranker_valid.json"}'):
            qid = item['q_id']
            item['pos_ids'] = each_qid_pid_dict[qid]
            # for top_i in item['top_n']:
            #     p_text = top_i[1].replace('百度经验:jingyan.baidu.com', '')
            #     top_i[1] = p_text
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_test_data(retrieve_test_data_path, out_file):
    """
    @param retrieve_test_data_path: 检索模型对测试集进行检索的top-n结果
    @param out_file: 输出路径
    """
    random.seed(1234)

    def split_data(data):
        """
        把数据切成4份
        """
        print(f'data len: {len(data)}')
        sub_size = int(len(data) / 4)
        for i in range(4):
            start_id = i * sub_size
            end_id = len(data) if i == 3 else start_id + sub_size
            sub_data = data[start_id:end_id]
            print(f'保存索引为 {start_id}:{end_id} 的数据到 {out_file}_{i}')
            with open(f'{out_file}_{i}', 'w', encoding='utf-8') as w:
                json.dump(sub_data, w, ensure_ascii=False, indent=4)

    print(f'加载 {retrieve_test_data_path}')
    with open(retrieve_test_data_path, 'r', encoding='utf-8') as f:
        # 数据格式：{'q_text': '问题', 'q_id': '问题ID', 'top_n': [['段落ID', '段落'],...]
        top_n_data = json.load(f)
    for item in tqdm(top_n_data, desc='去噪中'):
        top_n = item['top_n']
        for idx in range(len(top_n)):
            pid, ptext = top_n[idx][0], top_n[idx][1]
            # ptext = ptext.replace('百度经验:jingyan.baidu.com', '')
            if len(ptext) < 1:
                ptext = '&'
            top_n[idx][1] = ptext
    split_data(top_n_data)


def get_listwise_train_data_from_denoise_result(out_dir, origin_train_data_path='./train.json',
                                                is_random=True, n_samples=7):
    """
    @param out_dir: 训练集输出路径
    @param origin_train_data_path: 原始训练集
    @param is_random: 是否随机采样
    @param n_samples: 为每个问题采样负样本的个数
    @return:
    """
    random.seed(1234)

    def merge_and_save_denoise_result():
        bert_wwm_path = f'/home/chy/reranker-main/reranker_model/bert_wwm_pointwise_0.729/'
        nezha_wwm_path = f'/home/chy/reranker-main/reranker_model/nezha_wwm_pointwise_fgm0.01_0.737/'
        macbert_large_path = f'/home/chy/reranker-main/reranker_model/macbert_large_pointwise_0.740/'

        for i in range(4):
            train_data_top_n_with_scores = defaultdict(list)
            bert_top_n_data, nezha_top_n_data, macbert_top_n_data = defaultdict(dict), defaultdict(dict), defaultdict(
                dict)
            print(f'加载{bert_wwm_path}' + f'train_scores_{i}.json')
            with open(bert_wwm_path + f'train_scores_{i}.json', 'r') as f:
                sub_data = json.load(f)
                bert_top_n_data.update(sub_data)
            print(f'加载{nezha_wwm_path}' + f'train_scores_{i}.json')
            with open(nezha_wwm_path + f'train_scores_{i}.json', 'r') as f:
                sub_data = json.load(f)
                nezha_top_n_data.update(sub_data)
            print(f'加载{macbert_large_path}' + f'train_scores_{i}.json')
            with open(macbert_large_path + f'train_scores_{i}.json', 'r') as f:
                sub_data = json.load(f)
                macbert_top_n_data.update(sub_data)

            for q_id in tqdm(bert_top_n_data.keys(), desc='加载train_data_top_n_with_scores'):
                bert_top_n = bert_top_n_data[q_id]['top_n']
                nezha_top_n = nezha_top_n_data[q_id]['top_n']
                macbert_top_n = macbert_top_n_data[q_id]['top_n']
                temp_doc_list = []
                for bert, nezha, macbert in zip(bert_top_n, nezha_top_n, macbert_top_n):
                    score = (bert[2] + nezha[2] + macbert[2]) / 3
                    temp_doc_list.append((bert[0], bert[1], score))
                sorted_doc_list = sorted(temp_doc_list, key=lambda x: x[2], reverse=True)
                train_data_top_n_with_scores[q_id] = sorted_doc_list

            with open(f"sorted_from_merge_model_train_data_top200_{i}.json", 'w') as f:
                json.dump(train_data_top_n_with_scores, f, ensure_ascii=False)

    def read_merge_denoise_data():
        merge_denoise_data = {}
        for i in tqdm(range(4), desc='read_merge_denoise_data'):
            with open(f"sorted_from_merge_model_train_data_top200_{i}.json", 'r', encoding='utf-8') as f:
                merge_denoise_data.update(json.load(f))
        print(f'merge denoise data len {len(merge_denoise_data)}')
        return merge_denoise_data

    def read_qrel():
        nonlocal origin_train_data_path
        qrel = defaultdict(list)  # {qid:[pos_pids]}
        query_collection = defaultdict(str)  # {qid:q_text}
        doc_collection = defaultdict(str)  # {pid:p_text}
        with open(origin_train_data_path, 'r', encoding='utf-8') as f:
            # 数据格式：[{'question_id': '', 'question': '', 'answer_paragraphs': [{'paragraph_id': '', 'paragraph_text': ''}, ...]}]
            train_data = [json.loads(line) for line in f.readlines()]
        print(f'origin train query numbers: {len(train_data)}')
        for item in tqdm(train_data, desc=f'处理{origin_train_data_path}'):
            qid = item['question_id']
            query_collection[qid] = str(item['question'])
            for pos in item['answer_paragraphs']:
                pos_id = pos['paragraph_id']
                text = str(pos['paragraph_text'])
                if len(text) < 1:
                    print(f'skip {pos_id} {str(pos["paragraph_text"])}')
                    continue
                doc_collection[pos_id] = text
                qrel[qid].append(pos_id)
        return qrel, query_collection, doc_collection

    os.makedirs(out_dir, exist_ok=True)
    qrel, query_collection, doc_collection = read_qrel()
    # merge_and_save_denoise_result()
    denoise_data = read_merge_denoise_data()
    queries = list(qrel.keys())
    # 获取q的top n 召回段落（去除相关段落），以及将负样本段落添加到did对应的d_text对应字典。
    negs_pool = defaultdict(list)
    hard_negatives = defaultdict(list)
    no_judge = set()

    for qid in tqdm(denoise_data.keys(), desc='处理 merge_and_save_denoise_result'):
        count = 0
        for p in denoise_data[qid]:
            p_id, p_text, p_score = p[0], p[1], p[2]
            if p_id in qrel[qid]:
                continue
            if '￡是什么货币' in query_collection[qid] and len(p_text) < 10:
                continue
            if len(p_text) < 1:
                print(f'skip {query_collection[qid]} {p_id} {p[1]}')
                continue
            negs_pool[qid].append(p_id)
            doc_collection[p_id] = p_text
            if p_score < 0.1:
                hard_negatives[qid].append(p_id)
        #     else:
        #         count += 1
        # print(f'query {qid} have {count} > 0.9 negs.')

    print(f'{len(no_judge)} queries not judged and skipped', flush=True)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + '/reranker_train.json', 'w') as f:
        for qid in tqdm(queries, desc=f'构造{out_dir + "reranker_train.json"}'):
            # pick from top of the full initial ranking
            negs = negs_pool[qid]
            # shuffle if random flag is on
            if is_random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:n_samples] + hard_negatives[qid][:1]

            neg_dict = []
            for neg in negs:
                doc_text = doc_collection[neg]
                neg_dict.append({
                    'pid': neg,
                    'passage': doc_text if doc_text else '',
                })
            pos_dict = []
            for pos in qrel[qid]:
                doc_text = doc_collection[pos]
                pos_dict.append({
                    'pid': pos,
                    'passage': doc_text if doc_text else '',
                })
            query_dict = {
                'qid': qid,
                'query': query_collection[qid],
            }
            item_set = {
                'qry': query_dict,
                'pos': pos_dict,
                'neg': neg_dict,
            }
            f.write(json.dumps(item_set, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # 生成pointwise需要的训练/验证数据
    get_pointwise_train_data(retrieve_train_data_path='./train_data_top200.json')
    # 生成listwise需要的训练数据
    get_listwise_train_data(retrieve_train_data_path='/home/chy/GC-DPR-main/0.672/train_data_top200.json',
                            out_dir='./listwise_train_data/7_samples_seed2022/',
                            n_samples=7, seed=2022)
    # 生成listwise需要的验证数据
    get_listwise_valid_data(retrieve_dev_data_path='./dev_data_top50.json',
                            out_dir='./listwise_train_data/')
    # 生成测试数据
    get_test_data('./test_data_top50.json', './reranker_test_50')
    # 生成需要进行去噪的数据（原训练集）
    get_test_data('./train_data_top200.json', './train_data_top200')



