import numpy as np
import torch
from tqdm import tqdm

import data_reader
from evaluate import evaluate
from utils import config

def train_model(dataset, MODEL, target_size, padding_idx, n_fold=5, savename='model'):
    for fold in range(n_fold):
        train_data = dataset[dataset['kfold']!=fold]
        val_data = dataset[dataset['kfold']==fold]

        train_loader = data_reader.data_loader(train_data, config.BATCH_SIZE, shuffle=True)

        n_batches = int(np.ceil(len(train_data)/ config.BATCH_SIZE))

        cl_counts = dict(train_data.y.value_counts())
        threshold = 10
        cl_weight = np.ones(target_size)
        for y in cl_counts.keys():
            if cl_counts[y] >= 50:
                cl_weight[y] = 0.5
            else:
                cl_weight[y] = 1.

        model = MODEL(target_size, rv_comp=config.USE_RC, metadata=config.METADATA, padding_idx=padding_idx)
        model.to(config.DEVICE)

        loss_function = torch.nn.CrossEntropyLoss(torch.FloatTensor(cl_weight).cuda())
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-6)

        model.train()
        best_top10 = 0.
        best_epoch, no_improv = 0, 0
        stop_mt = False
        for epoch in range(config.MAX_EPOCH):
            total_loss, cnt = 0, 0
            pbar = tqdm(range(n_batches))
            for i in pbar:
                x, x_rc, md, y, x_len = next(iter(train_loader))
                score = model(x, x_rc, md)

                loss = loss_function(score, y)

                total_loss += loss.item()
                loss.backward()
                cnt += 1

                if cnt%config.GRAD_STEPS ==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.set_description("Epoch: {}, Loss: {:.4f}".format(epoch+1, total_loss/cnt))

            # print('Epoch: {}, Loss: {:.5f}'.format(epoch+1, total_loss/cnt))
            acc, top10, f1 = evaluate(val_data, model, target_size, loss_function=loss_function)
            if top10 > best_top10:
                print('*********** New best ***********')
                print('Acc: {:.4f}, top10: {:.4f}, F1: {:.4f}'.format(acc, top10, f1))
                best_top10 = top10
                best_epoch = epoch+1
                torch.save(model.state_dict(), config.MODEL_SAVE_DIR+'{}_{}.pt'.format(savename, fold+1))
                no_improv = 0
            else:
                no_improv += 1

            if no_improv > config.EARLY_STOP:
                break
        print('Fold {}, Best Validation: {} at Epoch: {}'.format(fold+1, best_top10, best_epoch))