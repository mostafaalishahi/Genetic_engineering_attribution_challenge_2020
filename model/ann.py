import torch
from torch import nn

class MODEL(nn.Module):
    def __init__(self, target_size, inpt_len=8000, hidden_size=256, rv_comp=True, metadata=True, padding_idx=0):
        super().__init__()
        self.rv_comp = rv_comp
        self.hidden_size = hidden_size
        self.dense = nn.Linear(inpt_len, self.hidden_size)
        self.act = nn.ReLU()
        if self.rv_comp:
            self.dense1 = nn.Linear((self.hidden_size*4*2)+39, 512)
        else:
            self.dense1 = nn.Linear((self.hidden_size*4)+39, 512)
        self.dense2 = nn.Linear(512, target_size)
        self.inp_dropout = nn.Dropout(0.05)
        self.dropout = nn.Dropout(0.3)

    def forward(self, sequence, sequence_rc, ft):
        sequence = self.inp_dropout(sequence)

        if self.rv_comp:
            sequence_rc = self.inp_dropout(sequence_rc)

        dense = self.dropout(self.act(self.dense(sequence)))

        if self.rv_comp:
            dense_rc = self.dropout(self.act(self.dense(sequence_rc)))
            comb = torch.cat([dense.reshape((dense.size(0), -1)), dense_rc.reshape((dense_rc.size(0), -1))], axis=-1)
        else:
            comb = dense.reshape((dense.size(0), -1))

        dense1 = self.act(self.dense1(torch.cat([comb, ft], axis=-1)))
        tag_scores = self.dense2(self.dropout(dense1))
        return tag_scores