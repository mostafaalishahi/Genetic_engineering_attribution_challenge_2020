import torch
from torch import nn

class MODEL(nn.Module):
    def __init__(self, target_size, n_filters=512, rv_comp=True, metadata=True, padding_idx=0):
        super().__init__()
        self.rv_comp = rv_comp
        self.metadata = metadata
        self.filters = n_filters
        self.cnn1 = nn.Conv1d(4, self.filters, kernel_size=12, stride=1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        if self.rv_comp:
            self.batchnorm1 = nn.BatchNorm1d(self.filters*2)
            if self.metadata:
                self.dense1 = nn.Linear((self.filters*2)+39, self.filters)
            else:
                self.dense1 = nn.Linear(self.filters*2, self.filters)
        else:
            self.batchnorm1 = nn.BatchNorm1d(self.filters)
            if self.metadata:
                self.dense1 = nn.Linear(self.filters+39, self.filters)
            else:
                self.dense1 = nn.Linear(self.filters, self.filters)

        self.activation = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(self.filters)
        self.hidden2tag = nn.Linear(self.filters, target_size)

        self.dropout = nn.Dropout(0.3)
        self.inp_dropout = nn.Dropout(0.05)

    def forward(self, sequence, sequence_rc, ft):
        sequence = self.inp_dropout(sequence)
        cnn1 = self.cnn1(sequence)
        maxpool = self.maxpool(cnn1).squeeze(-1)

        if self.rv_comp:
            sequence_rc = self.inp_dropout(sequence_rc)
            cnn1_rc = self.cnn1(sequence_rc)
            maxpool_rc = self.maxpool(cnn1_rc).squeeze(-1)

            bn1 = self.batchnorm1(torch.cat([maxpool, maxpool_rc], axis=-1))

        else:
            bn1 = self.batchnorm1(maxpool)

        dp1 = self.dropout(bn1)
        if self.metadata:
            dense1 = self.dense1(torch.cat([dp1, ft],axis=-1))
        else:
            dense1 = self.dense1(dp1)
        activation = self.activation(dense1)
        bn2 = self.batchnorm2(activation)
        dp2 = self.dropout(bn2)
        tag_scores = self.hidden2tag(dp2)
        return tag_scores