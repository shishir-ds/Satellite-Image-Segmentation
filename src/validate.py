import torch
from custom_loss_functions import *
import numpy
import numpy as np
from tqdm import tqdm
import logging


def validate_one_epoch(valData, model, criterion, device, val_loss=[],log_file=None):
    """
        Evaluate the model on separate Landsat scenes.
        Params:
            valData (DataLoader object) -- Batches of image chips from PyTorch custom dataset(AquacultureData)
            model -- Choice of segmentation Model.
            criterion -- Chosen function to calculate loss over validation samples.
            device (str): Either 'cuda' or 'cpu'.
            val_loss (empty list): To record average loss for each epoch
    """

    model.eval()
    path="/shishir/multi-temporal-crop-classification-baseline/output6/DataParallel_ep1000/training.log"
    
    def write_to_log(message):
        with open(path, 'a') as log_file:
            log_file.write(message + "\n")
            
    # mini batch iteration
    eval_epoch_loss = 0
    num_val_batches = len(valData)
    # print("num_val_batches:",num_val_batches)

    # if isinstance(model, torch.nn.DataParallel):
    #     model_device = next(model.module.parameters()).device
    #     # print("Model is on device:", model_device)
    
    with torch.no_grad():
        
        
        for ind, (img_chips, labels) in tqdm(enumerate(valData)):
            

            img = img_chips.to(device)
            label = labels.to(device)
            
            # print(model(img))
            pred = model(img)
            # print(np.unique(pred.cpu().numpy().flatten()),"*********************")

            # print("predictions:****",pred.shape)
            # print("label:*******",label.shape)
            
            loss = criterion(pred, label)

            eval_epoch_loss += loss.item()
            # print("validation loss per batch ",ind,":",loss.item())
    
    write_to_log('validation loss: {:.4f}'.format(eval_epoch_loss / num_val_batches))

    print('validation loss: {:.4f}'.format(eval_epoch_loss / num_val_batches))

    if val_loss is not None:
        val_loss.append(float(eval_epoch_loss / num_val_batches))