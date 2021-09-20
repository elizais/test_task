import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from model.Model import CNNModel
from preparatorCNN import *
from models.config import *


def train_model(model: CNNModel,
                loader: DataLoader,
                optimizer: torch.optim,
                criterion: nn.CrossEntropyLoss,
                epoch_num: int,
                device: torch.device,
                writer: torch.utils.tensorboard.writer.SummaryWriter,
                global_step: int,
                n_elem: int):
    model.train()
    model.is_training = True

    with tqdm(total=n_elem, desc=f'Epoch {epoch_num + 1}') as pbar:
        for batch, target in loader:
            batch = torch.tensor(batch, dtype=torch.float32, device=device)
            target = torch.tensor(target, dtype=torch.long, device=device)
            target = torch.squeeze(target, dim=1)

            optimizer.zero_grad()

            output = model(batch)
            print(output)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), global_step)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update(batch.shape[0])

            global_step += 1

    return global_step


def validate(model: CNNModel,
             loader: DataLoader,
             criterion: nn.CrossEntropyLoss,
             epoch_num: int,
             device: torch.device,
             n_elem: int):
    model.eval()
    model.is_training = False

    loss = 0
    with tqdm(total=n_elem, desc=f'Validation round {epoch_num+1}') as pbar:
        for elem_val, elem_true in loader:
            elem_val = torch.tensor(elem_val, dtype=torch.float32, device=device)
            elem_true = torch.tensor(elem_true, dtype=torch.long, device=device)
            elem_true = torch.squeeze(elem_true, dim=1)

            t, f, error_predict = 0, 0, 0
            with torch.no_grad():
                output = model(elem_val)
                for e_t, p_o, in zip(elem_true, output):
                    p_o = (p_o > 0.6).int()
                    ind = p_o.tolist().index(1) if 1 in p_o.tolist() else -1
                    if int(e_t) == ind and ind != -1:
                        t += 1
                    elif int(e_t) != ind and ind != -1:
                        f += 1
                    else:
                        error_predict += 1

                recall = t / (t + f)
                precision = t / (t + f + error_predict)
                loss += criterion(output, elem_true)
                pbar.update(elem_val.shape[0])
                pbar.set_postfix(**{'precision (batch)': precision},
                                 **{'recall (batch)': recall})

    score = loss / n_elem
    return score, precision, recall


if __name__ == '__main__':
    device_gpu = torch.device('cuda:0')

    train, train_test, validation, validation_test, test = prepare()

    batch_size = 200
    num_epochs = 250

    input_size = train.shape[-1]
    hidden_size = 200
    layer_size = 1
    output_size = 1
    learning_rate = 1e-3

    cnn_model = CNNModel()
    cnn_model.to(device=device_gpu)

    cross_entropy = nn.CrossEntropyLoss()

    train = TensorDataset(train, train_test)
    val = TensorDataset(validation, validation_test)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    n_train, n_val = len(train), len(val)

    writer_board = SummaryWriter()
    optim = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=5e-4)
    step = 0
    best_score = 0
    for epoch in range(num_epochs):

        step = train_model(cnn_model,
                           train_loader,
                           optim,
                           cross_entropy,
                           epoch,
                           device_gpu,
                           writer_board,
                           step,
                           n_train)

        val_score, precision_res, recall_res = validate(cnn_model,
                                                        val_loader,
                                                        cross_entropy,
                                                        epoch,
                                                        device_gpu,
                                                        n_val)
        if best_score < val_score:
            best_score = val_score
            if not os.path.exists(CHECKPOINT_DIR):
                os.makedirs(CHECKPOINT_DIR)
            filename = f'CNN__score_{val_score:.4f}__step_{step}.pth'
            torch.save(cnn_model.state_dict(), os.path.join(CHECKPOINT_DIR, filename))
        writer_board.add_scalar('score/val', val_score, epoch + 1)
        writer_board.add_scalar('estimates/precision', precision_res, epoch + 1)
        writer_board.add_scalar('estimates/recall', recall_res, epoch + 1)

    writer_board.close()