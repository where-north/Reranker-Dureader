"""
Name : convert_state_dict.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/5/18 18:15
Desc:
"""
import torch
from collections import OrderedDict

main_path = '/home/wangzhili/YangYang/reranker-main/reranker_model/2022-05-18_06_59_32/'

state_dict = torch.load(main_path + 'pytorch_model.bin')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[9:]
    new_state_dict[name] = v
torch.save(new_state_dict, main_path + 'pytorch_model.bin', _use_new_zipfile_serialization=False)