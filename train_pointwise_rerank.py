"""
Name : train_rerank_encoder.py
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
from torch.utils.data import DataLoader
from reranker_utils.optimizer_tools import *
from reranker_utils.utils import *
from reranker_utils.loss_tools import *
from reranker_utils.adversarial_tools import *
from reranker_utils.Datasets import FineTuneDataset
from reranker_model import Model
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

# 是否使用多GPU
if args.use_multi_gpu:
    torch.distributed.init_process_group(backend="nccl")
    if args.do_train and args.local_rank == 0:
        # 创建模型保存路径以及日志
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/finetune.log')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 创建模型保存路径以及日志
    if args.do_train:
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/finetune.log')


def input_to_device(tensor, args):
    if args.use_multi_gpu:
        return tensor.cuda(args.local_rank, non_blocking=True).long()
    else:
        return tensor.to(args.device).long()


def train(model):
    # 读取训练数据
    train_data = load_pontwise_data(args.train_data_path)
    valid_data = load_pontwise_data(args.valid_data_path)

    if args.use_multi_gpu:
        train_dataset = FineTuneDataset(train_data, args, model.get_tokenizer())
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8)
    else:
        train_dataset = FineTuneDataset(train_data, args, model.get_tokenizer())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = FineTuneDataset(valid_data, args, model.get_tokenizer())
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=8)

    criterion = nn.BCEWithLogitsLoss()
    saver = ModelSaver(args, patience=2, metric='max')
    optimizer, scheduler = build_optimizer(args, model, total_steps=len(train_loader) * args.epoch)

    if args.use_multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    if args.do_adversarial:
        adversary = build_adversary(args, model)
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
        for data in pbar:
            global_step += 1
            model.train()

            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            data['label'] = input_to_device(data['label'], args).float()

            if args.fp16:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), data['label'])
                scaler.scale(loss).backward()
                if args.do_adversarial:
                    backward_adversarial_loss(args, model, inputs, data['label'], adversary, scaler, criterion)
                # # Unscales the gradients of optimizer's assigned params in-place
                # scaler.unscale_(optimizer)
                # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), data['label'])
                loss.backward()

                if args.do_adversarial:
                    backward_adversarial_loss(args, model, inputs, data['label'], adversary, scaler, criterion)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if args.warmup: scheduler.step()
            if args.use_ema and epoch + 1 >= args.ema_start_epoch:
                ema.update()

            losses.append(loss.cpu().detach().numpy())
            output_array = torch.sigmoid(outputs.view(-1)).cpu().detach().numpy()
            output_array = [1 if i > 0.5 else 0 for i in output_array]
            label_array = data['label'].cpu().detach().numpy()
            acc_list.extend(output_array == label_array)
            pbar.set_description(
                f'epoch:{epoch + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

        if args.use_ema and epoch + 1 >= args.ema_start_epoch:
            ema.apply_shadow()

        if args.local_rank in [0, -1]:
            auc_score, accuracy, report = evaluate(model, valid_loader)
            early_stop = saver.save(accuracy, epoch, global_step, logger,
                                    model.module if args.local_rank != -1 else model)
            if early_stop:
                break
            logger.info(f'epoch:{epoch + 1}/{args.epoch}, valid auc_score: {auc_score}')
            logger.info(f'epoch:{epoch + 1}/{args.epoch}, vaild accuracy: {accuracy}')
            logger.info(f'{report}')


def evaluate(model, data_loader):
    model.eval()
    true, positive_logits, preds = [], [], []
    pbar = tqdm(data_loader, ncols=150)
    with torch.no_grad():
        for data in pbar:
            data['label'] = data['label'].float()
            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            outputs = model(inputs)
            positive_logit = torch.sigmoid(outputs.view(-1)).cpu().numpy()
            pred = [1 if i > 0.5 else 0 for i in positive_logit]
            true.extend(data['label'])
            positive_logits.extend(positive_logit)
            preds.extend(pred)

    auc_score = roc_auc_score(true, positive_logits)
    accuracy = accuracy_score(true, preds)
    report = classification_report(true, preds)

    return auc_score, accuracy, report


def predict(model):
    time_start = time()
    set_seed(args.seed)

    q_ids, doc_ids, q_texts, doc_texts, test_data = load_test_data(args.test_data_path, n_docs=args.n_docs)

    test_dataset = FineTuneDataset(test_data, args, model.get_tokenizer())
    test_loader = DataLoader(test_dataset, args.valid_batch_size, shuffle=False)

    model.eval()
    positive_logits = []
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=150):
            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            outputs = model(inputs)
            positive_logit = torch.sigmoid(outputs.view(-1)).cpu().numpy()
            positive_logits.extend(np.float_(positive_logit))

    print(f'predict data len: {len(positive_logits)}')
    result = defaultdict(dict)
    for i in range(len(q_ids)):
        if q_ids[i] in result:
            result[q_ids[i]]['top_n'].append((doc_ids[i], doc_texts[i], positive_logits[i]))
        else:
            result[q_ids[i]] = {'q_id': q_ids[i],
                                'q_text': q_texts[i],
                                'top_n': [(doc_ids[i], doc_texts[i], positive_logits[i])]}

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
        file_dir = args.pre_model_path + args.pre_model_timestamp
        file_list = os.listdir(file_dir)
        for name in file_list:
            if name == f'{args.model_type}.pth' or name.split('.')[-1] != 'pth':
                continue
            model_path = os.path.join(file_dir, name)
            if os.path.isfile(model_path) and name.split('-')[1] == f'epoch{args.pre_epoch}.pth':
                print('pretrain model: ', name)
                state_dict = torch.load(model_path, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device)
                train(model)
    elif args.do_predict:
        state_dict = torch.load(args.model_save_path + args.model_timestamp + f'/{args.model_type}_best_model.pth',
                                map_location='cuda')
        model.load_state_dict(state_dict, strict=True)
        model = model.to(args.device)
        predict(model)


if __name__ == '__main__':
    main()
