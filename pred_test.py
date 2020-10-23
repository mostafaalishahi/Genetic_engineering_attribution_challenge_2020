import os
import pandas as pd
import numpy as np
import torch

import data_reader
from evaluate import evaluate
from model.get_model import get_model
from utils import config

def get_predictions(data, model, target_size, savename, n_folds=5):
    result = np.zeros((len(data), target_size))

    for fold in range(n_folds):
        model.load_state_dict(torch.load(config.MODEL_SAVE_DIR+'{}_{}.pt'.format(savename, fold+1)))
        result += evaluate(data, model, target_size, return_pred=True)/n_folds

    return result

def write_predictions(data, result, labid_to_id, filename):
    id_to_labid = {v:k for k,v in labid_to_id.items()}
    result = pd.DataFrame(result, columns=[id_to_labid[i] for i in range(len(id_to_labid))])
    result.loc[:, 'sequence_id'] = data.index
    result = result.set_index('sequence_id')

    if not os.path.exists(config.PRED_SAVE_DIR):
        os.makedirs(config.PRED_SAVE_DIR)
    result.to_csv(config.PRED_SAVE_DIR+filename)

if __name__ == "__main__":
    data, labid_to_id, padding_idx = data_reader.load_data(config.DATA_DIR+'test_values.csv', train=False, val=False)

    #add dummy y_labels for data_loader
    data.loc[:, 'y'] = ''

    target_size = len(labid_to_id)
    print("Building model...")
    MODEL = get_model(config.MODEL)
    model = MODEL(target_size, rv_comp=config.USE_RC, metadata=config.METADATA, padding_idx=padding_idx)
    model.to(config.DEVICE)

    if config.MODEL_NAME_SUFFIX != '':
        savename = config.MODEL+'_'+config.MODEL_NAME_SUFFIX
    else:
        savename = config.MODEL

    result = get_predictions(data, model, target_size, savename, n_folds=config.N_FOLDS)
    write_predictions(data, result, labid_to_id, savename+'_test.csv')

    print(f'Test prediction file created as {savename}.csv')