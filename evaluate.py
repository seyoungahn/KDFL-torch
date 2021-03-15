""" Evaluate the model """

import argparse
import logging
import os

import numpy as np
import torch
from utils import *
import models.resnet_2 as resnet
import datautils as datautils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")

def evaluate(model, criterion, validloader, metrics, params):
    """
    Evaluate the model on 'num_steps' batches
    :param model: (torch.nn.Module) the neural network
    :param criterion: a function that takes y_pred and y_valid and computes the loss for the batch
    :param validloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    :param metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    :param params: (Params) hyperparameters
    :return:
    """
    # Set model to evaluation mode
    model.eval()

    # Summary for current eval loop
    summaries = []

    # Compute metrics over the dataset
    for x_valid, y_valid in validloader:
        # Move to GPU if available
        if params.cuda:
            x_valid, y_valid = x_valid.cuda(non_blocking=True), y_valid.cuda(non_blocking=True)
        # Fetch the next evaluation batch
        x_valid, y_valid = torch.autograd.Variable(x_valid), torch.autograd.Variable(y_valid)

        # Compute model output
        y_pred = model(x_valid)
        loss = criterion(y_pred, y_valid)

        # Extract data from torch Variable, move to CPU, convert to numpy arrays
        y_pred = y_pred.data.cpu().numpy()
        y_valid = y_valid.data.cpu().numpy()

        # Compute all metrics on this batch
        summary = {metric: metrics[metric](y_pred, y_valid) for metric in metrics}
        summary['loss'] = loss.item()
        summaries.append(summary)

    # Compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean

"""
This function duplicates "evaluate()" but ignores "criterion" simply for speed-up purpose.
Validation loss during KD model would display '0' all the time.
One can bring that info back by using the fetched teacher outputs during evaluation (refer to train.py)
"""
def evaluate_kd(model, validloader, metrics, params):
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
    for x_valid, y_valid in enumerate(validloader):
        # Move to GPU if available
        if params.cuda:
            x_valid, y_valid = x_valid.cuda(non_blocking=True), y_valid.cuda(non_blocking=True)
        # Fetch the next evaluation batch
        x_valid, y_valid = torch.autograd.Variable(x_valid), torch.autograd.Variable(y_valid)

        # Compute model output
        y_pred = model(x_valid)

        loss = 0.0 # Force validation loss to zero to reduce computation time

        # Extract data from torch Variable, move to cpu, convert to numpy arrays
        y_pred = y_pred.data.cpu().numpy()
        y_valid = y_valid.data.cpu().numpy()

        # Compute all metrics on this batch
        summary = {metric: metrics[metric](y_pred, y_valid) for metric in metrics}
        summary['loss'] = loss
        summaries.append(summary)

    # Compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]] for x in summaries) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean

# if __name__ == '__main__':
#     """
#         Evaluate the model on a dataset for one pass.
#     """
#     # Load the parameters
#     args = parser.parse_args()
#     json_path = os.path.join(args.model_dir, 'params.json')
#     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
#     params = utils.Params(json_path)
#
#     # use GPU if available
#     params.cuda = torch.cuda.is_available()     # use GPU is available
#
#     # Set the random seed for reproducible experiments
#     torch.manual_seed(230)
#     if params.cuda: torch.cuda.manual_seed(230)
#
#     # Get the logger
#     set_logger(os.path.join(args.model_dir, 'evaluate.log'))
#
#     # Create the input data pipeline
#     logging.info("Loading the dataset...")
#
#     # fetch dataloaders
#     # train_dl = data_loader.fetch_dataloader('train', params)
#     dev_dl = datautils.fetch_dataloader('dev', params)
#
#     logging.info("- done.")
#
#     # Define the model graph
#     model = resnet.ResNet().cuda() if params.cuda else resnet.ResNet()
#     optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
#
#     # fetch loss function and metrics
#     loss_fn_kd = loss_function_kd
#     metrics = resnet.metrics
#
#     logging.info("Starting evaluation...")
#
#     # Reload weights from the saved file
#     load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
#
#     # Evaluate
#     test_metrics = evaluate_kd(model, dev_dl, metrics, params)
#     save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
#     save_dict_to_json(test_metrics, save_path)