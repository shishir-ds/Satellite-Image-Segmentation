import torch
from torch.nn.modules.batchnorm import _BatchNorm
from custom_loss_functions import *
from custom_optimizer import SAM
from tqdm import tqdm
import logging


def disable_running_stats(model):
    """
    Disables the running statistics for Batch Normalization layers in the model.
    This function can be useful during certain training steps, 
    for example, in the first step of Sharpness-Aware Minimization (SAM) optimizer.
    """
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    """
    Enables the running statistics for Batch Normalization layers in the model.
    This function is typically used in pair with the 'disable_running_stats' function
    to re-enable batch normalization updates after they have been temporarily disabled,
    for example, in the second step of Sharpness-Aware Minimization (SAM) optimizer.
    """
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def train_one_epoch(trainData, model, criterion, optimizer, scheduler, device, train_loss=[],log_file=None):
    r"""
    Train the model.
    Arguments:
            trainData (DataLoader object) -- Batches of image chips from PyTorch custom dataset(AquacultureData)
            model (initialized model) -- Choice of segmentation Model to train.
            criterion -- Chosen function to calculate loss over training samples.
            optimizer -- Chosen function for optimization.
            scheduler -- Update policy for learning rate decay.
            device --(str) Either 'cuda' or 'cpu'.
            train_loss -- (empty list) To record average training loss for each epoch.
            
    """
    model.train()
    path="/shishir/multi-temporal-crop-classification-baseline/output6/DataParallel_ep1000/training.log"
    
    def write_to_log(message):
        with open(path, 'a') as log_file:
            log_file.write(message + "\n")

    # print(model,"^^^^^^^^^^^^^^")

    epoch_loss = 0
    num_train_batches = len(trainData)

    # print(len(trainData),"len train data")
    # print("device:",device)
    for img_chips, labels in tqdm(trainData):

        img = img_chips.to(device)
        label = labels.to(device)

        if isinstance(optimizer, SAM):

            enable_running_stats(model)

            def closure():
                predictions = model(img)
                loss = criterion(predictions, label)
                loss.mean().backward()
                return loss
            
            loss = criterion(model(img), label)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            if any(p.grad is not None for p in model.parameters()):
                optimizer.step(closure)
            
            disable_running_stats(model)

            # second forward-backward step
            def closure2():
                predictions = model(img)
                # print("predictions:****",predictions.shape)
                # print("label:*******",label.shape)
                loss = criterion(predictions, label)
                loss.mean().backward()
                return loss
            
            loss2 = criterion(model(img), label)
            # print("predictions:",model(img).shape)
            # print("label:",label.shape)
            
            loss2.mean().backward()
            optimizer.second_step(zero_grad=True)

            if any(p.grad is not None for p in model.parameters()):
                optimizer.step(closure2)
            
            epoch_loss += loss.item()
            # print("Train Loss per batch :",loss.item())

        else:
        
            model_out = model(img)
            loss = criterion(model_out, label)
        
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        isCyclicLR = False
        if type(scheduler) == torch.optim.lr_scheduler.CyclicLR:
            scheduler.step()
            isCyclicLR = True
    write_to_log('train loss:{}'.format(epoch_loss / num_train_batches))

    print('train loss:{}'.format(epoch_loss / num_train_batches))

    if isCyclicLR:
        print("LR: {}".format(scheduler.get_last_lr()))
        write_to_log("LR: {}".format(scheduler.get_last_lr()))

    if train_loss is not None:
        train_loss.append(float(epoch_loss / num_train_batches))