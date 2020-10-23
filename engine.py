import os
import argparse

import data_reader
from utils import config
from model.get_model import get_model
from train import train_model

config.set_seed(config.SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genome Attribution')
    parser.add_argument('--seed', type=int, dest='SEED', required=False)
    parser.add_argument('--data_dir', type=str, dest='DATA_DIR', required=False)
    parser.add_argument('--n_folds', type=int, dest='N_FOLDS', required=False)
    parser.add_argument('--model', type=str, dest='MODEL', required=False)
    parser.add_argument('--suffix', type=str, dest='MODEL_NAME_SUFFIX', required=False)
    parser.add_argument('--model_save_dir', type=str, dest='MODEL_SAVE_DIR', required=False)
    parser.add_argument('--pred_save_dir', type=str, dest='PRED_SAVE_DIR', required=False)
    parser.add_argument('--max_len', type=int, dest='MAX_LEN', required=False)
    parser.add_argument('--rc', type=bool, dest='USE_RC', required=False)
    parser.add_argument('--lr', type=float, dest='LEARNING_RATE', required=False)
    parser.add_argument('--bz', type=int, dest='BATCH_SIZE', required=False)
    parser.add_argument('--epoch', type=int, dest='MAX_EPOCH', required=False)
    parser.add_argument('--earlystop', type=int, dest='EARLY_STOP', required=False)
    parser.add_argument('--gradclip', type=float, dest='GRAD_CLIP', required=False)
    args = parser.parse_args()
    for k in vars(args):
        if getattr(args, k):
            setattr(config, k, getattr(args, k))

    # print({item:getattr(config, item) for item in dir(config) if not item.startswith("__")})
    print("Loading data....")
    data, labid_to_id, padding_idx = data_reader.load_data(config.DATA_DIR+'data.csv')
    target_size = len(labid_to_id)

    print(f'Using Device: {config.DEVICE}')

    print(f'Starting training of {config.MODEL}...')

    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)

    if config.MODEL_NAME_SUFFIX != '':
        savename = config.MODEL+'_'+config.MODEL_NAME_SUFFIX
    else:
        savename = config.MODEL
    train_model(data, get_model(config.MODEL), target_size, padding_idx, n_fold=config.N_FOLDS, savename=savename)