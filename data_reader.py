import warnings
warnings.filterwarnings("ignore")

import torch
import random
import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical

from utils import config

def get_ohe(batch, num_classes=5):
    ohe = to_categorical(batch, num_classes=num_classes)
    return ohe[:, :, :-1]

def get_input_tensor(x_batch, device):
    return torch.Tensor(get_ohe(x_batch)).to(device).transpose(1,2)

def data_loader(data, batch_size=config.BATCH_SIZE, shuffle=True, device=config.DEVICE):
    d = list(zip(data['sequence'], data['sequence_rc'], data['metadata'], data['y'], data['length']))
    while True:
        if shuffle:
            random.shuffle(d)

        x, x_rc, metadata, y, x_len = zip(*d)
        for i in range(0, len(x), batch_size):
            x_batch = get_input_tensor(x[i:i+batch_size], device=device)
            x_len_batch = torch.LongTensor(x_len[i:i+batch_size]).to(device)

            metadata_batch = torch.Tensor(metadata[i:i+batch_size]).to(device)
            y_batch = torch.LongTensor(y[i:i+batch_size]).to(device)

            if config.USE_RC:
                x_batch_rc = get_input_tensor(x_rc[i:i+batch_size], device=device)

                yield x_batch, x_batch_rc, metadata_batch, y_batch, x_len_batch

            else:
                yield x_batch, [], metadata_batch, y_batch, x_len_batch

def load_data(datafile, train=True, val=False):
    data = pd.read_csv(datafile)
    with open(config.DATA_DIR+'feat_label_ids.pkl', 'rb') as f:
        d = pickle.load(f)
    feat_col_to_id = d['feat_col_to_id']
    id_to_feat_col = d['id_to_feat_col']
    labid_to_id = d['labid_to_id']
    id_to_labid = d['id_to_labid']

    feat_col = [id_to_feat_col[i] for i in range(len(feat_col_to_id))]

    features = data[feat_col+['sequence_id']]
    features = features.set_index('sequence_id')
    features.loc[:, 'metadata'] = features[feat_col].apply(lambda row: row.values, axis=1)
    features = features.drop(feat_col, axis=1)

    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    rc_dict = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}

    if config.PAD_TOKEN not in char_to_int:
        char_to_int[config.PAD_TOKEN] = len(char_to_int)
    print('Feature dict: {}'.format(char_to_int))

    padding_idx = char_to_int[config.PAD_TOKEN]
    print(f'Padding index: {padding_idx}')

    if config.USE_RC:
        data.loc[:, 'seq1'] = data.apply(lambda x: [char_to_int[c] for c in x.sequence[:config.MAX_LEN]], axis=1)
        data.loc[:, 'seq2'] = data.apply(lambda x: [char_to_int[rc_dict[c]] for c in x.sequence[:config.MAX_LEN][::-1]], axis=1)


        data.loc[:, 'seq1'] = data.apply(lambda x: x.seq1[:config.MAX_LEN] if len(x.seq1)>=config.MAX_LEN else x.seq1 + [padding_idx]*(config.MAX_LEN-len(x.seq1)), axis=1)
        data.loc[:, 'seq2'] = data.apply(lambda x: x.seq2[:config.MAX_LEN] if len(x.seq2)>=config.MAX_LEN else x.seq2 + [padding_idx]*(config.MAX_LEN-len(x.seq2)), axis=1)

        data['length'] = data.apply(lambda x: min(len(x.sequence), config.MAX_LEN), axis=1)
        data.loc[:, 'sequence'] = data['seq1']
        data.loc[:, 'sequence_rc'] = data['seq2']

    else:
        print('Not using RC data')
        data.loc[:, 'seq1'] = data.apply(lambda x: [char_to_int[c] for c in x.sequence[:config.MAX_LEN]], axis=1)
        data['length'] = data.apply(lambda x: min(len(x.sequence), config.MAX_LEN), axis=1)

        data.loc[:, 'seq1'] = data.apply(lambda x: x.seq1[:config.MAX_LEN] if len(x.seq1)>=config.MAX_LEN else x.seq1 + [padding_idx]*(config.MAX_LEN-len(x.seq1)), axis=1)
        data.loc[:, 'sequence'] = data['seq1']
        data.loc[:, 'sequence_rc'] = ''

    if train or val:
        data = data[['sequence_id', 'sequence', 'sequence_rc', 'kfold', 'y', 'length']]
    else:
        data = data[['sequence_id', 'sequence', 'sequence_rc', 'length']]

    data = pd.merge(data, features, on='sequence_id')
    data = data.set_index('sequence_id')

    print(f'Data size: {len(data)}')

    return data, labid_to_id, padding_idx