import pandas as pd
import numpy as np
import pickle
import config
from sklearn import model_selection

def process_data():
    train_values = pd.read_csv(config.DATA_DIR+'train_values.csv')
    train_labels = pd.read_csv(config.DATA_DIR+'train_labels.csv')

    train_labels = train_labels.set_index('sequence_id')
    train_labels = pd.DataFrame(train_labels.idxmax(axis=1), columns=['lab_id'])

    feat_col = train_values.columns
    feat_col = train_values.columns
    feat_col = list(set(feat_col) - set(['sequence', 'sequence_id', 'kfold']))
    feat_col_to_id = {f:i for i, f in enumerate(feat_col)}
    id_to_feat_col = {v:k for k, v in feat_col_to_id.items()}
    labid_to_id = {k:v for v, k in enumerate(train_labels.lab_id.unique())}
    id_to_labid = {v:k for k, v in labid_to_id.items()}

    d = {'feat_col_to_id': feat_col_to_id,
        'id_to_feat_col': id_to_feat_col,
        'labid_to_id': labid_to_id,
        'id_to_labid': id_to_labid
        }
    with open(config.DATA_DIR+'feat_label_ids.pkl', 'wb') as f:
        pickle.dump(d, f)

    train_labels['y'] = train_labels.apply(lambda x: labid_to_id.get(x.lab_id), axis=1)
    train_labels = train_labels[['y']]
    train_labels = train_labels.reset_index()

    train_values['length'] = train_values.apply(lambda x: len(x.sequence), axis=1)

    data = pd.merge(train_values, train_labels, on='sequence_id')
    data = data.reset_index(drop=True)

    data['kfold'] = -1
    kf = model_selection.KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    for fold, (trn_, val_) in enumerate(kf.split(X=data, y=data['y'])):
        data.loc[val_, 'kfold'] = fold

    data.to_csv(config.DATA_DIR+'data.csv')

if __name__ == "__main__":
    config.set_seed(config.SEED)
    process_data()