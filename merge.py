"""
Name : merge.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/5/26 18:26
Desc:
"""
import json
from tqdm import tqdm
from collections import defaultdict


def norm_score(score_list):
    min_, max_ = min(score_list), max(score_list)
    diff = max_ - min_
    norm_score_list = []
    for sco in score_list:
        if diff > 0:
            norm_score_list.append((sco - min_) / diff)
        else:
            norm_score_list.append(1)
    return norm_score_list


def get_reranker_res(model_path):
    main_path = f'/home/chy/reranker-main/reranker_model/{model_path}/'
    print(f'加载{main_path}')
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
        # sort_top_n = sorted(top_n, key=lambda x: x[2], reverse=True)
        pid_list = [i[0] for i in top_n]
        score_list = [i[2] for i in top_n]
        norm_score_list = norm_score(score_list)
        sort_res[q_id] = [[i, j] for i, j in zip(pid_list, norm_score_list)]
    return sort_res


if __name__ == '__main__':

    merge_result = {}
    pointwise_sort_res1 = get_reranker_res(model_path='nezha_wwm_pointwise_fgm0.01_0.737')
    pointwise_sort_res2 = get_reranker_res(model_path='bert_wwm_pointwise_0.729')
    pointwise_sort_res3 = get_reranker_res(model_path='macbert_large_pointwise_0.740')
    sort_res1 = get_reranker_res(model_path='macbert_large_0.773_0.787')
    sort_res2 = get_reranker_res(model_path='0.771_0.784')

    for qid in sort_res1.keys():
        pid_score_dict = {}
        for pid_score1, pid_score2, pid_score3, pid_score4, pid_score5 in zip(sort_res1[qid], sort_res2[qid],
                                                                              pointwise_sort_res1[qid],
                                                                              pointwise_sort_res2[qid],
                                                                              pointwise_sort_res3[qid]):
            pid1, score1 = pid_score1[0], pid_score1[1]
            pid2, score2 = pid_score2[0], pid_score2[1]
            pid3, score3 = pid_score3[0], pid_score3[1]
            pid4, score4 = pid_score4[0], pid_score4[1]
            pid5, score5 = pid_score5[0], pid_score5[1]
            assert pid1 == pid2
            pid_score_dict[pid1] = (score1 + score2 + (score3 + score4 + score5) / 3) / 3

        merge_pid_scores = [i for i in pid_score_dict.items()]
        merge_pid_scores = sorted(merge_pid_scores, key=lambda x: x[1], reverse=True)
        merge_top50_pids = [i[0] for i in merge_pid_scores]
        merge_result[qid] = merge_top50_pids

    with open('./merge_result.json', 'w', encoding='utf-8') as f:
        json.dump(merge_result, f, indent=4, ensure_ascii=False)
