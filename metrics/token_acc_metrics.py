# -*- coding:utf-8 -*-
# create: @time: 9/20/23 11:52
import torch


class TokenAccMetric:
    def __init__(self, pad_token_id=0, eos_token_id=2, **kwargs):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.total_tokens = 0
        self.token_correct = 0
        self.token_acc = []

    def add(self, tgt_seqs, preds):
        shape_diff = preds.shape[1] - tgt_seqs.shape[1]
        if shape_diff < 0:
            preds = torch.nn.functional.pad(preds, (0, -shape_diff), "constant", self.pad_token_id)
        elif shape_diff > 0:
            tgt_seqs = torch.nn.functional.pad(tgt_seqs, (0, shape_diff), "constant", self.pad_token_id)
        mask = torch.logical_or(tgt_seqs != self.pad_token_id, preds != self.pad_token_id)
        tok_acc = (preds == tgt_seqs)[mask].float()
        self.token_acc.append(tok_acc.mean().item())
        self.token_correct += int(tok_acc.sum().item())
        self.total_tokens += len(tok_acc)

    def mean(self):
        return self.token_correct / self.total_tokens
