"""
Name : adversarial_tools.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2021/9/29 14:57
Desc:
"""
import torch
from torch.cuda.amp import autocast


def build_adversary(args, model):
    adversary = None
    if args.adversarial_type == 'fgm':
        adversary = FGM(args, model)
    elif args.adversarial_type == 'pgd':
        adversary = PGD(args, model)
    return adversary


def backward_adversarial_loss(args, model, inputs, label, adversary, scaler, criterion=None):
    if args.adversarial_type == 'fgm':
        adversary.attack()
        if args.fp16:
            with autocast():
                # 模型内部没有定义损失函数
                if criterion:
                    outputs_adv = model(inputs)
                    outputs_adv = outputs_adv.view(-1) if args.wise == 'pointwise' else outputs_adv
                    loss_adv = criterion(outputs_adv, label)
                # 模型内部定义了损失函数
                else:
                    loss_adv = model(**inputs)[0]
                loss_adv = loss_adv / args.accumulation_steps
            scaler.scale(loss_adv).backward()
        else:
            # 模型内部没有定义损失函数
            if criterion:
                outputs_adv = model(inputs)
                outputs_adv = outputs_adv.view(-1) if args.wise == 'pointwise' else outputs_adv
                loss_adv = criterion(outputs_adv, label)
            # 模型内部定义了损失函数
            else:
                loss_adv = model(**inputs)[0]
            loss_adv = loss_adv / args.accumulation_steps
            loss_adv.backward()
        adversary.restore()
    elif args.adversarial_type == 'pgd':
        K = args.pgd_adv_k
        adversary.backup_grad()
        for t in range(K):
            adversary.attack(is_first_attack=(t == 0))
            if t != K - 1:
                model.zero_grad()
            else:
                adversary.restore_grad()
            if args.fp16:
                with autocast():
                    if criterion:
                        outputs_adv = model(inputs)
                        outputs_adv = outputs_adv.view(-1) if args.wise == 'pointwise' else outputs_adv
                        loss_adv = criterion(outputs_adv, label)
                    else:
                        loss_adv = model(**inputs)[0]
                    loss_adv = loss_adv / args.accumulation_steps
                scaler.scale(loss_adv).backward()
            else:
                if criterion:
                    outputs_adv = model(inputs)
                    outputs_adv = outputs_adv.view(-1) if args.wise == 'pointwise' else outputs_adv
                    loss_adv = criterion(outputs_adv, label)
                else:
                    loss_adv = model(**inputs)[0]
                loss_adv = loss_adv / args.accumulation_steps
                loss_adv.backward()
        adversary.restore()


class FGM:
    def __init__(self, args, model):
        self.model = model
        self.backup = {}
        self.epsilon = args.fgm_epsilon

    def attack(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.emb_name = 'word_embeddings'
        self.epsilon = args.pgd_epsilon
        self.alpha = args.pgd_alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
