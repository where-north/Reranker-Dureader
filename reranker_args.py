import argparse

parser = argparse.ArgumentParser()

# 微调参数
parser.add_argument('--gpu_id', default=1, type=int,
                    help="使用的GPU id")
parser.add_argument("--q_maxlen", default=32, type=int,
                    help="问题最长句长")
parser.add_argument("--p_head_maxlen", default=384, type=int,
                    help="段落头部最长句长")
parser.add_argument("--p_tail_maxlen", default=0, type=int,
                    help="段落尾部最长句长")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--model_save_path", default='./reranker_model/', type=str,
                    help="微调模型保存路径")
parser.add_argument("--model_timestamp", default='2022-04-23_03_56_41', type=str,
                    help="微调模型保存时间戳")
parser.add_argument("--device", default='cuda', type=str,
                    help="使用GPU")
parser.add_argument("--do_train", action='store_true', default=False,
                    help="是否微调")
parser.add_argument("--do_train_with_encoder", action='store_true', default=False,
                    help="是否微调")
parser.add_argument("--do_predict", action='store_true', default=False,
                    help="是否预测")
parser.add_argument("--do_predict_with_encoder", action='store_true', default=False,
                    help="是否预测")
parser.add_argument("--do_train_after_pretrain", action='store_true', default=False,
                    help="是否预训练后再微调")
parser.add_argument("--warmup", action='store_true', default=False,
                    help="是否采用warmup学习率策略")
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument("--pre_model_path", default='./pretrain_model/', type=str,
                    help="预训练模型保存路径")
parser.add_argument("--pre_model_timestamp", default='2021-08-09_11_05_24', type=str,
                    help="预训练模型保存时间戳")
parser.add_argument("--lr", default=1e-5, type=float,
                    help="初始学习率")
parser.add_argument("--encoder_lr", default=1e-5, type=float,
                    help="encoder初始学习率")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--epoch", default=5, type=int,
                    help="训练轮次")
parser.add_argument('--seed', type=int, default=42,
                    help="随机种子")
parser.add_argument("--batch_size", default=32, type=int,
                    help="train_batch size")
parser.add_argument("--valid_batch_size", default=32, type=int,
                    help="valid batch size")
parser.add_argument("--use_ema", action='store_true', default=False,
                    help="是否使用指数加权平均")
parser.add_argument('--ema_start_epoch', default=2, type=int,
                    help="第几个epoch使用EMA")
parser.add_argument("--do_adversarial", action='store_true', default=False,
                    help="是否使用对抗训练")
parser.add_argument("--adversarial_type", default='fgm', type=str, choices=['fgm', 'pgd'],
                    help="对抗训练类型")
parser.add_argument("--fgm_epsilon", default=0.2, type=float)
parser.add_argument('--pgd_adv_k', type=int, default=10)
parser.add_argument('--pgd_alpha', type=float, default=0.3)
parser.add_argument('--pgd_epsilon', type=float, default=0.5)
parser.add_argument("--num_classes", default=1, type=int,
                    help="类别数目")
parser.add_argument("--pre_epoch", default=-1, type=int,
                    help="选取哪个epoch的预训练模型")
parser.add_argument("--model_type", default='roberta', type=str,
                    help="预训练模型类型")
parser.add_argument("--train_data_path", default='./data/reranker_train.tsv', type=str)
parser.add_argument("--valid_data_path", default='./data/reranker_valid.tsv', type=str)
parser.add_argument("--test_data_path", default='./data/base_data4rerank.json', type=str)
parser.add_argument("--use_multi_gpu", action='store_true', default=False)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--max_grad_norm', type=float, default=-1)
parser.add_argument("--fp16", action='store_true', default=False,
                    help="是否使用混合精度训练")
parser.add_argument('--n_docs', type=int, default=50, help="每个问题召回的文档数量")
parser.add_argument('--valid_n_docs', type=int, default=50, help="对验证集的top50进行重排序")
parser.add_argument('--wise', type=str, help="训练方式", choices=['pointwise', 'pairwise', 'listwise'])
parser.add_argument('--train_group_size', type=int, default=8, help="listwise 组内段落数量(1+n)")
parser.add_argument('--alpha', type=float, default=0.5, help='celoss权重（alpha）bceloss权重（1-alpha）')
parser.add_argument('--accumulation_steps', type=int, default=1, help="梯度累积步数")
parser.add_argument('--start_save_steps', type=int, default=30000, help="开始保存模型的步数")
parser.add_argument('--save_steps', type=int, default=5000, help="每隔多少步保存模型")
parser.add_argument("--use_grad_cached", action='store_true', default=False,
                    help="是否使用梯度缓存")
parser.add_argument("--chunk_size", default=8, type=int, help="梯度缓存子批量的大小，需要被batch_size*train_group_size整除")
parser.add_argument("--output_type", default='pooler', type=str, choices=['pooler', 'mean_pooling', 'cls', 'dynamic'])

args = parser.parse_args()

# 开源预训练模型路径
model_map = dict()

model_map['roberta'] = {
    'model_path': '/home/yy/pretrainModel/chinese_roberta_wwm_ext_pytorch/pytorch_model.bin',
    'config_path': '/home/yy/pretrainModel/chinese_roberta_wwm_ext_pytorch/config.json',
    'vocab_path': '/home/yy/pretrainModel/chinese_roberta_wwm_ext_pytorch/vocab.txt'}
model_map['nezha_wwm'] = {'model_path': '/home/yy/pretrainModel/nezha-cn-wwm/pytorch_model.bin',
                          'config_path': '/home/yy/pretrainModel/nezha-cn-wwm/config.json',
                          'vocab_path': '/home/yy/pretrainModel/nezha-cn-wwm/vocab.txt'}
model_map['nezha_base'] = {'model_path': '/home/yy/pretrainModel/nezha-cn-base/pytorch_model.bin',
                           'config_path': '/home/yy/pretrainModel/nezha-cn-base/config.json',
                           'vocab_path': '/home/yy/pretrainModel/nezha-cn-base/vocab.txt'}
model_map['ernie-gram'] = {'model_path': '/home/yy/pretrainModel/ernie-gram-zh/pytorch_model.bin',
                           'config_path': '/home/yy/pretrainModel/ernie-gram-zh/config.json',
                           'vocab_path': '/home/yy/pretrainModel/ernie-gram-zh/vocab.txt'}
model_map['bert_wwm'] = {'model_path': '/home/yy/pretrainModel/bert_wwm/pytorch_model.bin',
                         'config_path': '/home/yy/pretrainModel/bert_wwm/config.json',
                         'vocab_path': '/home/yy/pretrainModel/bert_wwm/vocab.txt'}
model_map['macbert_large'] = {'model_path': '/home/chy/MODEL/chinese-macbert-large/pytorch_model.bin',
                              'config_path': '/home/chy/MODEL/chinese-macbert-large/config.json',
                              'vocab_path': '/home/chy/MODEL/chinese-macbert-large/vocab.txt'}
model_map['macbert_base'] = {'model_path': '/home/yy/pretrainModel/chinese-macbert-base/pytorch_model.bin',
                             'config_path': '/home/yy/pretrainModel/chinese-macbert-base/config.json',
                             'vocab_path': '/home/yy/pretrainModel/chinese-macbert-base/vocab.txt'}
model_map['roberta_large'] = {'model_path': '/home/yy/pretrainModel/roberta-large/pytorch_model.bin',
                              'config_path': '/home/yy/pretrainModel/roberta-large/config.json',
                              'vocab_path': '/home/yy/pretrainModel/roberta-large/vocab.txt'}
model_map['nezha_large_wwm'] = {'model_path': '/home/chy/MODEL/nezha-large-wwm/pytorch_model.bin',
                                'config_path': '/home/chy/MODEL/nezha-large-wwm/config.json',
                                'vocab_path': '/home/chy/MODEL/nezha-large-wwm/vocab.txt'}
model_map['nezha_large_base'] = {'model_path': '/home/chy/MODEL/nezha-large-base/pytorch_model.bin',
                                 'config_path': '/home/chy/MODEL/nezha-large-base/config.json',
                                 'vocab_path': '/home/chy/MODEL/nezha-large-base/vocab.txt'}
model_map['bert_large'] = {'model_path': '/home/yy/pretrainModel/bert_large_chinese/pytorch_model.bin',
                           'config_path': '/home/yy/pretrainModel/bert_large_chinese/config.json',
                           'vocab_path': '/home/yy/pretrainModel/bert_large_chinese/vocab.txt'}
