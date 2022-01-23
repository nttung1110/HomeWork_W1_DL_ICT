from model import ThreeLayerMLP
from dataset import TwoDimRecSamplingData

import yaml
import wandb
import pdb
import torch.nn as nn
import torch
import torch.optim as optim


def load_config(path_config):
    with open(path_config, 'r') as fp:
        cfg = yaml.safe_load(fp)

    cfg_train = {}
    for key, value in cfg.items():
        cfg_train[key] = value
    
    return cfg_train

def val_on_epoch(model, val_data, cfg_train, criterion, optimizer):
    model.eval()

    result_all_batch = None
    all_label = None

    for index, data in enumerate(val_data, 0):
        data_points, labels = data
        preds = model(data_points)

        if result_all_batch is None:
            result_all_batch = preds

        else:
            # concatenate for later computation
            result_all_batch = torch.cat((result_all_batch, preds), 0)

        if all_label is None:
            all_label = labels
        else:
            all_label = torch.cat((all_label, labels), 0)

    # accuracy
    result_all_batch = result_all_batch.argmax(dim=1).type(torch.FloatTensor).unsqueeze(dim=1)
    all_label = all_label.type(torch.FloatTensor).unsqueeze(dim=1)
    val_acc = torch.sum(torch.eq(result_all_batch, all_label)).item() / result_all_batch.shape[0]
    
    return val_acc


def trainval(model, train_data, val_data, cfg_train, criterion, optimizer):
    num_epoch = 100
    # training 
    freq_print = 100
    for epoch in range(num_epoch):
        running_loss = 0.0
        model.train()
        for index, data in enumerate(train_data, 0):
            data_points, labels = data
            labels = labels.type(torch.LongTensor)
            bs = data_points.shape[0]

            optimizer.zero_grad()
            preds = model(data_points)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if index % freq_print == 0 and index!=0:
            #     # print every freq_print
            #     print("[epoch_%d, batch_%d] loss: %.10f" % (epoch, index, running_loss / freq_print))
            #     wandb.log({'loss': running_loss/freq_print})
            #     running_loss = 0.0

        # print every epoch
        acc_val_epoch = val_on_epoch(model, val_data, cfg_train, criterion, optimizer)
        print("[epoch_%d], loss: %.10f, accuracy on test: %.10f" % (epoch, running_loss/bs, acc_val_epoch))

        # postfix as experiment name        
        postfix = '_'.join(['3_hidden_layer_use_drop_out', str(cfg_train['use_dropout'])])
        
        wandb.log({'loss_'+postfix: running_loss/bs})

        wandb.log({'accuracy_'+postfix: acc_val_epoch})

if __name__ == "__main__":
    path_config = '/home/nttung/DL_Homework/config/multiple_layer_exp.yaml'
    cfg_train = load_config(path_config)

    # dataset 
    train_data = TwoDimRecSamplingData(100000, "train")
    val_data = TwoDimRecSamplingData(20000, "val")
    test_data = TwoDimRecSamplingData(20000, "test")


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512,
                                                shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=512, num_workers=2)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, num_workers=2)

    if cfg_train["use_dropout"] is False:
        model = ThreeLayerMLP(2)
    else:
        model = ThreeLayerMLP(2, cfg_train["drop_out_rate"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    wandb.init(project='Official_v4_HomeworkDL', entity='nttung1110')
    wandb.watch(model, log_freq=100)

    print("Training with configuration:", cfg_train)

    trainval(model, train_loader, val_loader, cfg_train, criterion, optimizer)
