import os
import pandas as pd
import numpy as np
import torch

from evaluate import evaluate
import data_reader
from utils.metrics import print_all_metrics
from model.get_model import get_model
from utils import config

def get_predictions(data, model, labid_to_id, n_folds=5, savename='result'):
    columns = ['sequence_id', 'kfold', 'y', 'lab_id'] + list(labid_to_id.keys())
    result = pd.DataFrame(columns=columns)
    result = result.set_index('sequence_id')
    id_to_labid = {v: k for k, v in labid_to_id.items()}
    target_size = len(labid_to_id)
    for fold in range(n_folds):
        val = data[data['kfold']==fold]
        y_true = val['y'].astype(int)
        model.load_state_dict(torch.load(config.MODEL_SAVE_DIR+'{}_{}.pt'.format(savename, fold+1)))
        y_probs = evaluate(val, model, target_size, return_pred=True)
        temp_df = pd.DataFrame({k: y_probs[:, v] for k, v in labid_to_id.items()})
        temp_df.loc[:, 'sequence_id'] = val.index
        temp_df = temp_df.set_index('sequence_id')
        temp_df.loc[:, 'kfold'] = val['kfold']
        temp_df.loc[:, 'y'] = val['y']
        temp_df.loc[:, 'lab_id'] = [id_to_labid[v] for v in val['y']]
        result = result.append(temp_df)

        print(f'Fold: {fold+1}')
        print_all_metrics(y_true, y_probs, target_size=target_size)

    y_true = result['y'].astype(int)
    y_probs = result[[id_to_labid[i] for i in range(target_size)]].values

    print(f'Overall:')
    print_all_metrics(y_true, y_probs, target_size=target_size)

    if not os.path.exists(config.PRED_SAVE_DIR):
        os.makedirs(config.PRED_SAVE_DIR)

    to_save = config.PRED_SAVE_DIR+'{}_val.csv'.format(savename)
    print(f'Saving predictions to {to_save}')
    result.to_csv(to_save)

if __name__ == "__main__":
    # print({item:getattr(config, item) for item in dir(config) if not item.startswith("__")})
    print("Loading data....")
    data, labid_to_id, padding_idx = data_reader.load_data(config.DATA_DIR+'data.csv', train=False, val=True)
    target_size = len(labid_to_id)

    print("Building model...")
    MODEL = get_model(config.MODEL)
    model = MODEL(target_size, rv_comp=config.USE_RC, metadata=config.METADATA, padding_idx=padding_idx)
    model.to(config.DEVICE)

    if config.MODEL_NAME_SUFFIX != '':
        savename = config.MODEL+'_'+config.MODEL_NAME_SUFFIX
    else:
        savename = config.MODEL
    get_predictions(data, model, labid_to_id, n_folds=config.N_FOLDS, savename=savename)