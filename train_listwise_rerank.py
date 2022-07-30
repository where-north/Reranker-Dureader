"""
Name : train_listwise_rerank.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/22 15:39
Desc:
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reranker_args import args
from time import time
from tqdm import tqdm
from reranker_utils.optimizer_tools import *
from reranker_utils.utils import *
from reranker_utils.loss_tools import *
from reranker_utils.adversarial_tools import *
from reranker_utils.Datasets import FineTuneDataset, FineTuneDatasetForListWise
from reranker_model import Model, Model2, Model3
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy

# 是否使用多GPU
if args.use_multi_gpu:
    torch.distributed.init_process_group(backend="nccl")
    if not (args.do_predict or args.do_predict_with_encoder) and args.local_rank == 0:
        # 创建模型保存路径以及日志
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/finetune.log')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 创建模型保存路径以及日志
    if not (args.do_predict or args.do_predict_with_encoder):
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/finetune.log')


def input_to_device(tensor, args):
    if args.use_multi_gpu:
        return tensor.cuda(args.local_rank, non_blocking=True).long()
    else:
        return tensor.to(args.device).long()


def train(model):
    # 读取训练数据
    train_data = load_listwise_train_data(args)

    if args.use_multi_gpu:
        train_dataset = FineTuneDatasetForListWise(train_data, args, model.get_tokenizer())
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  collate_fn=FineTuneDatasetForListWise.collate, num_workers=8)
    else:
        train_dataset = FineTuneDatasetForListWise(train_data, args, model.get_tokenizer())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=FineTuneDatasetForListWise.collate, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    saver = ModelSaver(args, patience=5, metric='max')
    optimizer, scheduler = build_optimizer(args, model, total_steps=len(train_loader) * args.epoch)

    if args.use_multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    model.zero_grad()

    scaler = None
    if args.fp16: scaler = GradScaler()
    global_step = 0

    for epoch in range(args.epoch):
        if args.use_multi_gpu:
            train_sampler.set_epoch(epoch)
        if args.use_ema and epoch + 1 >= args.ema_start_epoch:
            ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)
        pbar = tqdm(train_loader, ncols=150)
        losses, acc_list = [], []
        for i, data in enumerate(pbar):
            global_step += 1
            model.train()

            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            bs = int(data['input_ids'].size()[0] / args.train_group_size)
            inputs['label'] = input_to_device(torch.tensor([0 for _ in range(bs)]), args).long()

            if args.use_grad_cached:
                loss, outputs = do_forward_with_grad_cached(args, model, copy.deepcopy(inputs), criterion, scaler)
            else:
                loss, outputs = do_forward(args, model, copy.deepcopy(inputs), criterion)
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if args.warmup:
                    scheduler.step()

            if args.use_ema and epoch + 1 >= args.ema_start_epoch:
                ema.update()

            losses.append(loss.cpu().detach().numpy())
            output_array = torch.softmax(outputs, dim=-1).cpu().detach().numpy()
            label_array = inputs['label'].cpu().detach().numpy()
            acc_list.extend(np.argmax(output_array, axis=-1) == label_array)
            pbar.set_description(
                f'epoch:{epoch + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses) * args.accumulation_steps:.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

            if args.local_rank in [0,
                                   -1] and global_step % args.save_steps == 0 and global_step > args.start_save_steps:
                mrr = evaluate(model, args)
                early_stop = saver.save(mrr, epoch, global_step, logger,
                                        model.module if args.local_rank != -1 else model)
                if early_stop:
                    break
                logger.info(f'step:{global_step}, valid mrr: {mrr}')
        if args.use_ema and epoch + 1 >= args.ema_start_epoch:
            ema.apply_shadow()

        if args.local_rank in [0, -1]:
            mrr = evaluate(model, args)
            early_stop = saver.save(mrr, epoch, global_step, logger, model.module if args.local_rank != -1 else model)
            if early_stop:
                break
            logger.info(f'epoch:{epoch + 1}/{args.epoch}, valid mrr: {mrr}')


def do_forward(args, model, inputs, criterion):
    label = inputs.pop('label')
    if args.fp16:
        with autocast():
            outputs = model(inputs)
    else:
        outputs = model(inputs)

    loss = criterion(outputs, label)
    loss = loss / args.accumulation_steps

    return loss, outputs


def do_forward_with_grad_cached(args, model, inputs, criterion, scaler):
    label = inputs.pop('label')
    input_ids_chunks = inputs['input_ids'].split(args.chunk_size)
    attention_mask_chunks = inputs['attention_mask'].split(args.chunk_size)
    token_type_ids_chunks = inputs['token_type_ids'].split(args.chunk_size)

    all_outputs = []

    for input_ids_chunk, attention_mask_chunk, token_type_ids_chunk in zip(
            input_ids_chunks, attention_mask_chunks, token_type_ids_chunks):
        _input = {
            "input_ids": input_ids_chunk,
            "attention_mask": attention_mask_chunk,
            "token_type_ids": token_type_ids_chunk
        }
        with torch.no_grad():
            if args.fp16:
                with autocast():
                    outputs = model(_input)
            else:
                outputs = model(_input)
        all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs)
    all_outputs = all_outputs.float().detach().requires_grad_()
    batch_size, group_size = int(all_outputs.size()[0] / args.train_group_size), args.train_group_size
    output = model.encoder(all_outputs.view(batch_size, group_size, -1))[0]
    all_logits = model.classifier(model.dropout(output))
    all_logits = all_logits.view(
        -1,
        args.train_group_size
    )
    loss = criterion(all_logits, label)
    loss = loss / args.accumulation_steps
    loss.backward()
    output_grads = all_outputs.grad.split(args.chunk_size)

    for input_ids_chunk, attention_mask_chunk, token_type_ids_chunk, grad in zip(
            input_ids_chunks, attention_mask_chunks, token_type_ids_chunks, output_grads):
        _input = {
            "input_ids": input_ids_chunk,
            "attention_mask": attention_mask_chunk,
            "token_type_ids": token_type_ids_chunk
        }
        if args.fp16:
            with autocast():
                outputs = model(_input)
                surrogate = torch.dot(outputs.flatten().float(), grad.flatten())
        else:
            outputs = model(_input)
            surrogate = torch.dot(outputs.flatten().float(), grad.flatten())

        if args.fp16:
            scaler.scale(surrogate).backward()
        else:
            surrogate.backward()
    for q_model_name, q_para in model.named_parameters():
        if 'hf_model' in q_model_name and q_model_name != 'hf_model.bert.embeddings.word_embeddings.weight' and q_para.grad is not None:
            q_para.grad /= (args.batch_size * args.train_group_size / args.chunk_size)

    return loss, all_logits


def evaluate(model, args):
    """
    计算MRR@10
    @param model:
    @param args:
    @return: MRR@10
    """
    p_ids, pos_ids, valid_data = load_listwise_valid_data(args)
    valid_dataset = FineTuneDataset(valid_data, args, model.get_tokenizer())
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)
    model.eval()
    logits = []
    pbar = tqdm(valid_loader, ncols=150)
    with torch.no_grad():
        for data in pbar:
            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            outputs = model(inputs)
            logits.extend(outputs.view(-1).cpu().detach().numpy())

    logits = np.expand_dims(np.array(logits).reshape(len(pos_ids), args.valid_n_docs), axis=-1)
    p_ids = np.expand_dims(np.array(p_ids), axis=-1)
    p_ids_with_logit = np.concatenate([p_ids, logits], axis=-1).tolist()
    reciprocal_rank = 0
    for idx, item in enumerate(p_ids_with_logit):
        sort_item = sorted(item, key=lambda x: x[1], reverse=True)
        top_10_pids = [i[0] for i in sort_item[:10]]
        for rank, pid in enumerate(top_10_pids):
            if pid in pos_ids[idx]:
                reciprocal_rank += 1 / (rank + 1)
                break

    return reciprocal_rank / len(pos_ids)


def predict(model):
    time_start = time()
    set_seed(args.seed)

    q_ids, doc_ids, q_texts, doc_texts, test_data = load_test_data(args.test_data_path, n_docs=args.n_docs)

    test_dataset = FineTuneDataset(test_data, args, model.get_tokenizer())
    test_loader = DataLoader(test_dataset, args.valid_batch_size, shuffle=False, num_workers=8)

    model.eval()
    logits = []
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=150):
            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            outputs = model(inputs)
            logit = outputs.view(-1).cpu().numpy()
            logits.extend(np.float_(logit))

    print(f'predict data len: {len(logits)}')
    result = defaultdict(dict)
    for i in range(len(q_ids)):
        if q_ids[i] in result:
            result[q_ids[i]]['top_n'].append((doc_ids[i], doc_texts[i], logits[i]))
        else:
            result[q_ids[i]] = {'q_text': q_texts[i],
                                'top_n': [(doc_ids[i], doc_texts[i], logits[i])]}

    filedir, filename = os.path.split(args.test_data_path)
    prefix = filename.split('_')[0]
    data_id = filename.split('_')[-1]
    out_file = args.model_save_path + args.model_timestamp + f"/{prefix}_scores_{data_id}.json"

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    time_end = time()
    print(f'finish {time_end - time_start}s')


def main():
    set_seed(args.seed)

    model = Model.from_pretrained(args=args)

    if args.do_train:
        if args.use_multi_gpu:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
        else:
            model.to(args.device)

        train(model)
    elif args.do_train_after_pretrain:
        model = Model2.from_pretrained(args=args)
        file_dir = args.pre_model_path + args.pre_model_timestamp
        file_list = os.listdir(file_dir)
        for name in file_list:
            if name == f'{args.model_type}.pth' or name.split('.')[-1] != 'pth':
                continue
            model_path = os.path.join(file_dir, name)
            if os.path.isfile(model_path) and 'best' in name:
                print('best finetune model: ', name)
                state_dict = torch.load(model_path, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device)
                for w_name, para in model.named_parameters():
                    if 'hf_model' in w_name:
                        para.requires_grad = False
                train(model)
    elif args.do_predict:

        state_dict = torch.load(args.model_save_path + args.model_timestamp + f'/{args.model_type}_best_model.pth',
                                map_location='cuda')
        model.load_state_dict(state_dict, strict=True)
        model = model.to(args.device)
        predict(model)
    elif args.do_train_with_encoder:
        model = Model3.from_pretrained(args=args)
        if args.use_multi_gpu:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
        else:
            model.to(args.device)

        train(model)
    elif args.do_predict_with_encoder:
        model = Model3.from_pretrained(args=args)
        state_dict = torch.load(args.model_save_path + args.model_timestamp + f'/{args.model_type}_best_model.pth',
                                map_location='cuda')
        model.load_state_dict(state_dict, strict=True)
        model = model.to(args.device)
        predict(model)


if __name__ == '__main__':
    main()
