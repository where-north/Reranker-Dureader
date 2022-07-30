"""
Name : reranker_model.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2021/8/16 20:06
Desc:
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.nn.functional as F
import torch.nn as nn
import torch
from reranker_args import model_map
from transformers import BertTokenizer
from collections import OrderedDict
from transformers.models.bert.modeling_bert import BertModel, BertForSequenceClassification, BertPreTrainedModel, \
    BertLayer
from nezha_model_utils.nezha.modeling_nezha import NeZhaModel, NeZhaForSequenceClassification, NeZhaPreTrainedModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from nezha_model_utils.nezha.configuration_nezha import NeZhaConfig

MODEL_CONFIG = {'nezha_wwm': 'NeZhaConfig', 'nezha_base': 'NeZhaConfig', 'roberta': 'RobertaConfig',
                'ernie-gram': 'RobertaConfig', 'bert_wwm': 'RobertaConfig', 'macbert_large': 'RobertaConfig',
                'macbert_base': 'RobertaConfig', 'roberta_large': 'RobertaConfig', 'nezha_large_wwm': 'NeZhaConfig',
                'nezha_large_base': 'NeZhaConfig', 'bert_large': 'RobertaConfig', }
MODEL_NAME = {'nezha_wwm': 'NeZhaForSequenceClassification', 'nezha_base': 'NeZhaForSequenceClassification',
              'roberta': 'BertForSequenceClassification', 'bert_wwm': 'BertForSequenceClassification',
              'ernie-gram': 'BertForSequenceClassification', 'macbert_large': 'BertForSequenceClassification',
              'macbert_base': 'BertForSequenceClassification', 'roberta_large': 'BertForSequenceClassification',
              'nezha_large_wwm': 'NeZhaForSequenceClassification', 'nezha_large_base': 'NeZhaForSequenceClassification',
              'bert_large': 'BertForSequenceClassification', }


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MyBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class MyNeZha(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = NeZhaModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        return outputs


class MyBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )


class Model(nn.Module):
    def __init__(self, model, args):
        super(Model, self).__init__()
        self.hf_model = model
        self.args = args

    def get_tokenizer(self):
        return BertTokenizer(
            vocab_file=model_map[self.args.model_type]['vocab_path'],
            do_lower_case=True)

    def forward(self, x):
        output = self.hf_model(**x, return_dict=True)
        logits = output.logits
        if self.args.wise == 'listwise' and self.training:
            logits = logits.view(
                -1,
                self.args.train_group_size
            )

        return logits

    @classmethod
    def from_pretrained(cls, args):
        bert_config = globals()[MODEL_CONFIG[args.model_type]].from_json_file(model_map[args.model_type]['config_path'])
        bert_config.num_labels = args.num_classes
        model = globals()[MODEL_NAME[args.model_type]](config=bert_config)
        state_dict = torch.load(model_map[args.model_type]['model_path'])
        if args.model_type == 'ernie-gram':
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'bert.' + k
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        model.load_state_dict(new_state_dict, strict=False)

        return cls(model, args)


class Model2(nn.Module):
    def __init__(self, model, args):
        super(Model2, self).__init__()
        self.hf_model = model
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)
        self.args = args

    def get_tokenizer(self):
        return BertTokenizer(
            vocab_file=model_map[self.args.model_type]['vocab_path'],
            do_lower_case=True)

    def forward(self, x):
        output = self.hf_model(**x)[1]
        output = self.fc1(output)
        output = torch.relu(output)
        logits = self.fc2(output)
        if self.args.wise == 'listwise' and self.training:
            logits = logits.view(
                -1,
                self.args.train_group_size
            )

        return logits

    @classmethod
    def from_pretrained(cls, args):
        bert_config = globals()[MODEL_CONFIG[args.model_type]].from_json_file(model_map[args.model_type]['config_path'])
        bert_config.num_labels = args.num_classes
        # model = MyNeZha(config=bert_config)
        model = MyBert(config=bert_config)
        state_dict = torch.load(model_map[args.model_type]['model_path'])
        if args.model_type == 'ernie-gram':
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'bert.' + k
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        model.load_state_dict(new_state_dict, strict=False)

        return cls(model, args)


class Model3(nn.Module):
    def __init__(self, model, args):
        super(Model3, self).__init__()
        self.hf_model = model
        self.config = RobertaConfig.from_json_file('./encoder_config.json')
        self.linear = nn.Linear(768, self.config.hidden_size)
        self.encoder = MyBertEncoder(config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        if args.output_type == 'dynamic':
            self.fc1 = nn.Linear(768, 768)
            self.fc2 = nn.Linear(768, 1)
            self.ln = nn.LayerNorm(768)
        self.args = args

    def get_tokenizer(self):
        return BertTokenizer(
            vocab_file=model_map[self.args.model_type]['vocab_path'],
            do_lower_case=True)

    def forward(self, x):
        output = None
        if self.args.output_type == 'pooler':
            output = self.hf_model(**x)[1]
        elif self.args.output_type == 'mean_pooling':
            output = mean_pooling(self.hf_model(**x), x['attention_mask'])
        elif self.args.output_type == 'cls':
            output = self.hf_model(**x)[0][:, 0, :]
        elif self.args.output_type == 'dynamic':
            output = self.hf_model(**x)[2]
            output = self.get_dym_layer(output)[:, 0, :]
            output = self.ln(self.fc1(output))
        output = self.linear(output)
        if self.args.use_grad_cached:
            if self.training:
                return output
            else:
                batch_size, group_size = 1, self.args.valid_batch_size
                output = self.encoder(output.view(batch_size, group_size, -1))[0]
                logits = self.classifier(self.dropout(output))
                return logits.squeeze(-1)
        else:
            if self.training:
                batch_size, group_size = int(output.size()[0] / self.args.train_group_size), self.args.train_group_size
            else:
                batch_size, group_size = 1, self.args.valid_batch_size
            output = self.encoder(output.view(batch_size, group_size, -1))[0]
            logits = self.classifier(self.dropout(output))
            if self.args.wise == 'listwise' and self.training:
                logits = logits.view(
                    -1,
                    self.args.train_group_size
                )

            elif not self.training:
                logits = logits.squeeze(-1)

            return logits

    @classmethod
    def from_pretrained(cls, args):
        bert_config = globals()[MODEL_CONFIG[args.model_type]].from_json_file(model_map[args.model_type]['config_path'])
        bert_config.num_labels = args.num_classes
        if args.output_type == 'dynamic':
            bert_config.output_hidden_states = True
        if 'bert' in args.model_type:
            model = MyBert(config=bert_config)
        elif 'nezha' in args.model_type:
            model = MyNeZha(config=bert_config)
        state_dict = torch.load(model_map[args.model_type]['model_path'])
        if args.model_type == 'ernie-gram':
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'bert.' + k
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        model.load_state_dict(new_state_dict, strict=False)

        return cls(model, args)

    def get_dym_layer(self, all_encoder_layers):
        layer_logits = []
        all_encoder_layers = all_encoder_layers[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.fc2(layer))
        layer_logits = torch.cat(layer_logits, 2)
        layer_dist = torch.softmax(layer_logits, dim=-1)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        output = torch.squeeze(output, 2)

        return output


class ModelForDynamicLen(nn.Module):
    def __init__(self, bert_config, args):
        super(ModelForDynamicLen, self).__init__()
        MODEL_NAME = {'nezha_wwm': 'NeZhaModel', 'nezha_base': 'NeZhaModel', 'roberta': 'BertModel'}
        self.bert = globals()[MODEL_NAME[args.model_type]](config=bert_config)
        self.args = args

        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)

        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])

    def forward(self, input_ids):
        output = None
        if self.args.struc == 'cls':
            output = torch.stack(
                [self.bert(input_id.to(self.args.device))[0][0][0]
                 for input_id in input_ids])

        if self.args.AveragePooling:
            output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)

        # output = self.dropout(output)
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        return output
