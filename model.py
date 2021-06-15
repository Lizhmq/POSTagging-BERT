import os
import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel, BertModel
from torch.autograd import Variable


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class POSModel(nn.Module):

    def __init__(self, hidden, classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.classes = classes
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, classes)
        self.bert = BertModel.from_pretrained(self.args.pretrain_dir)
        self._init_cls_weight()


    def _init_cls_weight(self, initializer_range=0.02):
        for layer in (self.classifier, ):
            layer.weight.data.normal_(mean=0.0, std=initializer_range)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
        torch.save(self.classifier.state_dict(), os.path.join(path, "cls.bin"))
        
    def from_pretrained(self, path):
        self.bert = BertModel.from_pretrained(path)
        self.classifier.load_state_dict(torch.load(os.path.join(path, "cls.bin"), map_location=self.device))
        return self

    def _sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.range(0, max_len - 1).long().to(sequence_length.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        seq_length_expand = (sequence_length.unsqueeze(1)
                            .expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def predict(self, input_ids, input_mask, sts, ends, lens):
        sequence_output = self.bert(input_ids, input_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        st_shape = sts.shape
        sts = sts.unsqueeze(-1).expand(list(st_shape) + [768])
        selected = torch.gather(sequence_output, dim=1, index=sts)
        logits = self.classifier(selected)
        preds = torch.argmax(logits, dim=-1)
        return preds        

    def acc(self, input_ids, input_mask, sts, ends, lens, y):
        preds = self.predict(input_ids, input_mask, sts, ends, lens, y)
        # seq_mask = self._sequence_mask(lens)
        correct = torch.sum((preds == y).float())
        return correct.sum() / lens.sum()


    def forward(self, input_ids, input_mask, sts, ends, lens, y=None):
        sequence_output = self.bert(input_ids, input_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        st_shape = sts.shape
        sts = sts.unsqueeze(-1).expand(list(st_shape) + [768])
        # print(sts.shape)
        selected = torch.gather(sequence_output, dim=1, index=sts)
        # print(selected.shape)
        # accumulate_hiddens = normalize_hiddens(sequence_output, ses, ends)
        # assert(accumulate_hiddens.shape[1] == ses.shape[0])
        # !!!! accumulate, not average
        # div = (ends - ses).unsqueeze(-1)
        # accumulate_hiddens = accumulate_hiddens / div
        # logits = self.classifier(accumulate_hiddens)
        
        logits = self.classifier(selected)
        # print(logits.shape)

        if y == None:
            return logits
        else:
            celoss = nn.CrossEntropyLoss(ignore_index=-100)
            logits = logits.reshape([-1, self.classes])
            y = y.reshape([-1])
            loss = celoss(logits, y)
            # logits_flat = logits.view(-1, logits.size(-1))
            # log_probs_flat = F.log_softmax(logits_flat)
            # target_flat = y.view(-1, 1)
            # losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
            # losses = losses_flat.view(*y.size())
            # # mask: (batch, max_len)
            # seq_mask = self._sequence_mask(lens)
            # losses = losses * seq_mask.float()
            # loss = losses.sum() / lens.float().sum()
            return loss



def build_model(args, classes, load_path=None):
    model = POSModel(768, classes, args, args.device)
    if load_path is not None:
        model = model.from_pretrained(load_path).to(args.device)
    return model