import argparse
import logging
import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import utils
import models.resnet_2 as resnet
import datautils

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_dir', default='experiments/baseline_KD', help='Directory containing params.json')
# parser.add_argument('--restore_file', default=None, help='Optional, name of the file in --model_dir containing weights to reload before training') ## 'best' or 'train'
#
## Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, criterion_kd, trainloader, metrics, params):
    """
    Train the model on 'num_steps' batches
    :param model: (torch.nn.Module) the neural network
    :param optimizer: (torch.optim) optimizer for parameters of model
    :param metrics: (dict)
    :param params: (Params) hyperparameters
    """

    # Set model to training mode
    model.train()
    teacher_model.eval()

    # Summary for current training loop and a running average object for loss
    summaries = []
    avg_loss = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(trainloader)) as t:
        for batch_idx, (x_train, y_train) in enumerate(trainloader):
            # Move to GPU if available
            if params.cuda:
                x_train, y_train = x_train.cuda(non_blocking=True), y_train.cuda(non_blocking=True)

            # Convert to torch Variable
            x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

            # Compute model output, fetch teacher output, and compute KD loss
            y_pred = model(x_train)

            # Get one batch output from teacher_knowledgies list
            with torch.no_grad():
                teacher_knowledge = teacher_model(x_train)
            if params.cuda:
                teacher_knowledge = teacher_knowledge.cuda(non_blocking=True)

            loss = criterion_kd(y_pred, y_train, teacher_knowledge, params)

            # Clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if batch_idx % params.save_summary_steps == 0:
                # Extract data from torch Variable, move to CPU, convert to numpy arrays
                y_pred = y_pred.data.cpu().numpy()
                y_train = y_train.data.cpu().numpy()

                # Compute all metrics on this batch
                summary = {metric:metrics[metric](y_pred, y_train) for metric in metrics}
                summary['loss'] = loss.item()
                summaries.append(summary)

            # Update the avearge loss
            avg_loss.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
            t.update()

    # Compute mean of all metrics in summary
    # print(summaries)
    metrics_mean = {metric:np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean

def evaluate_kd(model, teacher_model, criterion, validloader, metrics, params):
    """
    Evaluate the model on 'num_steps' batches.
    :param model: (torch.nn.Module) the neural network
    :param validloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    :param metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    :param params: (Params) hyperparameters
    num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # Set model to evaluation mode
    model.eval()

    # Summary for current eval loop
    summaries = []

    # Compute metrics over the dataset
    for batch_idx, (x_valid, y_valid) in enumerate(validloader):
        # Move to GPU if available
        if params.cuda:
            x_valid, y_valid = x_valid.cuda(non_blocking=True), y_valid.cuda(non_blocking=True)
        # Fetch the next evaluation batch
        x_valid, y_valid = torch.autograd.Variable(x_valid), torch.autograd.Variable(y_valid)

        # Compute model output
        y_pred = model(x_valid)

        with torch.no_grad():
            teacher_knowledge = teacher_model(x_valid)
        if params.cuda:
            teacher_knowledge = teacher_knowledge.cuda(non_blocking=True)

        loss = criterion(y_pred, y_valid, teacher_knowledge, params)
        # loss = 0.0 # Force validation loss to zero to reduce computation time

        # Extract data from torch Variable, move to cpu, convert to numpy arrays
        y_pred = y_pred.data.cpu().numpy()
        y_valid = y_valid.data.cpu().numpy()

        # Compute all metrics on this batch
        summary = {metric: metrics[metric](y_pred, y_valid) for metric in metrics}
        summary['loss'] = loss.item()
        summaries.append(summary)

    # Compute mean of all metrics in summary
    # print("Evaluation summaries")
    # print(summaries)
    metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean

def train_and_evaluate_kd(model, teacher_model, trainloader, validloader, optimizer, criterion_kd, metrics, params, model_dir, restore_file=None):
    """
    Train the model and evaluate every epoch
    :param model: (torch.nn.Module) the neural network
    :param teacher_model: (Params) hyperparameters
    :param model_dir: (string) directory containing config, weights and log
    :param restore_file: (string) - file to restore (without its extension .ptr.tar)
    """
    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_valid_acc = 0.0

    # TensorBoard logger setup
    # board_logger = utils.Board_logger(os.path.join(model_dir, 'board_logs'))

    # Learning rate scedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "cnn_distill":
        # For cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(params.num_epochs):
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))

        # Compute number of batches in one epoch (one full pass over the training set
        train_kd(model, teacher_model, optimizer, criterion_kd, trainloader, metrics, params)

        # Evaluate for one epoch on validation set
        valid_metrics = evaluate_kd(model, validloader, metrics, params)

        valid_acc = valid_metrics['accuracy']
        is_best = valid_acc >= best_valid_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_valid_acc = valid_acc

            # Save best valid metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_valid_best_weights.json")
            utils.save_dict_to_json(valid_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_valid_last_weights.json")
        utils.save_dict_to_json(valid_metrics, last_json_path)

        #============ TensorBoard logging: uncomment below to turn in on ============#
        # # (1) Log the scalar values
        # info = {
        #     'valid accuracy': valid_acc
        # }

        # for tag, value in info.items():
        #     board_logger.scalar_summary(tag, value, epoch+1)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace(',', '/')
        #     board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     # board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)