import pandas as pd
import numpy as np
import pickle

from utils import config

def create_submission_file(input_files, weights, id_to_labid):
    columns = [id_to_labid[i] for i in range(len(id_to_labid))]

    for i, f in enumerate(input_files):
        result = pd.read_csv(config.PRED_SAVE_DIR+f, index_col='sequence_id')
        if i ==0:
            probs = np.array(result[columns])*weights[i]
        else:
            probs += np.array(result[columns])*weights[i]

    #get top10 predictions and assign equal probability to them. This reduces the final file size. If accuracy metric is desired, please change this block of code
    top10_index = np.argpartition(probs, -10)[:, -10:]
    top10 = np.zeros((len(result), probs.shape[1]))
    for i in range(len(top10)):
        top10[i][top10_index[i]] = 0.1

    final_df = pd.DataFrame(top10, columns=columns)
    final_df.loc[:, 'sequence_id'] = result.index
    final_df = final_df.set_index('sequence_id')

    submission_format = pd.read_csv(config.DATA_DIR+'submission_format.csv', index_col='sequence_id')
    for col in submission_format.columns:
        if col not in final_df.columns:
            print(f'{col} not found in results file')
            final_df.loc[:, col] = 0.

    final_df = final_df[submission_format.columns]

    assert submission_format.shape == final_df.shape
    assert (final_df.columns == submission_format.columns).all()

    savename = config.PRED_SAVE_DIR+'submission.csv'
    print(f'Saving submission file to {savename}')
    final_df.to_csv(savename)

if __name__ == "__main__":
    with open(config.DATA_DIR+'feat_label_ids.pkl', 'rb') as f:
        d = pickle.load(f)
    id_to_labid = d['id_to_labid']

    if config.MODEL_NAME_SUFFIX != '':
        filename = config.MODEL+'_'+config.MODEL_NAME_SUFFIX
    else:
        filename = config.MODEL

    files = [filename+'_test.csv']
    weights = [1.0]
    create_submission_file(files, weights, id_to_labid)
