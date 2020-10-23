import torch
from tqdm import tqdm
import numpy as np

import data_reader
from utils.metrics import top10_accuracy_scorer, accuracy, f1_score
from utils import config

@torch.no_grad()
def evaluate(dataset, model, target_size, return_pred=False, loss_function=None):
    results = []

    eval_loader = data_reader.data_loader(dataset, config.BATCH_SIZE, shuffle=False)
    n_batches = int(np.ceil(len(dataset)/config.BATCH_SIZE))
    pbar = tqdm(range(n_batches))
    model.eval()
    total_loss = 0.
    for i in pbar:
        x, x_rc, md, y, x_len = next(iter(eval_loader))
        score = model(x, x_rc, md)
        loss = 0.
        if loss_function is not None:
            loss = loss_function(score, y)
            total_loss += loss.mean().item()
        probs = torch.nn.Softmax(dim=1)(score)
        results.append(probs.detach().cpu().numpy())
    model.train()

    results = np.vstack(results)

    if loss_function is not None:
        print('Val Loss: {:.5f}'.format(total_loss/n_batches))

    if return_pred:
        return results

    y_pred = np.argmax(results, axis=1)
    acc = accuracy(dataset['y'], y_pred)
    top10 = top10_accuracy_scorer(np.array(dataset['y']), results, target_size=target_size)
    f1 = f1_score(dataset['y'], y_pred)
    # print(acc, top10, f1)
    return acc, top10, f1