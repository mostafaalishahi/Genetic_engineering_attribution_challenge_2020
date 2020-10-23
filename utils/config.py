import random
import torch
import numpy as np

def set_seed(seed):
    print(f'Setting random seed to {seed}')
    #random seed
    random.seed(seed)

    #numpy seed
    np.random.seed(seed)

    #torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 123
DATA_DIR = './data/'
N_FOLDS = 5
#model to use ['ann', 'cnn']
MODEL = 'cnn'
#suffix to add to model names while saving
MODEL_NAME_SUFFIX = ''
#location to save model files
MODEL_SAVE_DIR = './models/'
#location to save results
PRED_SAVE_DIR = './preds/'
#maximum length of sequence
MAX_LEN = 8000
#PAD_TOKEN
PAD_TOKEN = 'N'
#if to use Reverse_Complement or not
USE_RC = True
#if to use metadata
METADATA = True

#Training hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCH = 50
EARLY_STOP = 10
GRAD_CLIP = 1
GRAD_STEPS = 1