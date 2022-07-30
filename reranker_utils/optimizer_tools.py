"""
Name : optimizer_tools.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2021/9/29 15:02
Desc:
"""
import torch
from torch.optim import Optimizer
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup


def build_optimizer(args, model, total_steps):
    bert_param_optimizer = []
    other_param_optimizer = []

    for n, p in model.named_parameters():
        if 'hf_model' in n:
            bert_param_optimizer.append((n, p))
        else:
            other_param_optimizer.append((n, p))

    no_decay = ['bias', 'LayerNorm.weight']

    if other_param_optimizer:

        optimizer_grouped_parameters = [
            # hf_model
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay,
             "lr": args.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0,
             "lr": args.lr},
            # other_layer
            {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay,
             "lr": args.encoder_lr},
            {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0,
             "lr": args.encoder_lr},
        ]

    else:

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                  eps=args.adam_epsilon)

    scheduler = None
    if args.warmup:
        total_steps = int(total_steps / args.accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

    return optimizer, scheduler


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
