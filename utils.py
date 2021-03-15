import json
import logging
import os
import shutil
import torch
import errno
from collections import OrderedDict

# import tensorflow as tf
import numpy as np
import scipy.misc

import torch.nn as nn
import torch.nn.functional as F

try:
    from StringIO import StringIO   # Python 2.7
except ImportError:
    from io import BytesIO          # Python 3.x


class Params():
    """
    Class that loads hyperparameters from a json file.
    Example:
        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        ## Loads parameters from json file
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        ## Gives dict-like access to Params instance by 'params.dict['learning_rate']
        return self.__dict__

class RunningAverage():
    """
    A simple class that maintains the running average of a quantity
    Example:
        avg_loss = RunningAverage()
        avg_loss.update(2)
        avg_loss.update(4)
        avg_loss() = 3
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file 'log_path'.

    In general, it is useful to have a logger so that every output to the terminal is saved in a permanent file.
    Here we save it to 'model_dir/train.log'.
    Example:
        logging.info("Start training...")
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    :param state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
    :param is_best: (bool) True if it is the best model seen till now
    :param checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint directory does not exist: making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint directory exists.")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    :param checkpoint: (string) filename which needs to be loaded
    :param model: (torch.nn.Module) model for which the parameters are loaded
    :param optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("Checkpoint file does not exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # This helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def loss_function(outputs, labels):
    """
    Compute the cross-entropy loss given outputs and labels
    :param outputs: (Variable) dimension batch_size x 6 - output of the model
    :param labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    :return: loss (Variable) cross-entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions.
          This example demonstrates how you can easily define a custom loss function
    """
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_function_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    Hyperparameters: temperature, alpha
    Note: the KL Divergence for PyTorch comparing the softmaxs of teacher and student expects the input tensor to be log probabilities
    """
    alpha = params.alpha
    T = params.temperature
    # KLDivergence issue: reduction='mean' doesn't return the true KL divergence value
    #                     please use reduction = 'batchmean' which aligns with KL math definition.
    #                     In the next major release, 'mean' will be changed to be the same as 'batchmean'
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    :param outputs: (np.ndarray) output of the model
    :param labels: (np.ndarray) [0, 1, ..., num_classes-1]
    :return: (float) accuracy in [0, 1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(labels.size)

def mkdir_p(path):
    ''' make dir if not exist '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# class TensorBoardLogger(object):
#     """
#     TensorBoard log utility
#     """
#
#     def __init__(self, log_dir):
#         ## Create a summary writer logging to log_dir.
#         self.writer = tf.summary.FileWriter(log_dir)
#
#     def scalar_summary(self, tag, value, step):
#         ## Log a scalar variable.
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)
#
#     def image_summary(self, tag, images, step):
#         ## Log a list of images.
#         img_summaries = []
#         for i, img in enumerate(images):
#             # Write the image to a string
#             try:
#                 s = StringIO()
#             except:
#                 s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")
#
#             # Create an Image object
#             img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
#
#             # Create a Summary value
#             img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
#
#         # Create and write Summary
#         summary = tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)
#
#     def histo_summary(self, tag, values, step, bins=1000):
#         ## Log a histogram of the tensor of values.
#
#         # Create a histogram using numpy
#         counts, bin_edges = np.histogram(values, bins=bins)
#
#         # Fill the fields of the histogram proto
#         hist = tf.HistogramProto()
#         hist.min = float(np.min(values))
#         hist.max = float(np.max(values))
#         hist.num = int(np.prod(values.shape))
#         hist.sum = float(np.sum(values))
#         hist.sum_squares = float(np.sum(values**2))
#
#         # Drop the start of the first bin
#         bin_edges = bin_edges[1:]
#
#         # Add bin edges and counts
#         for edge in bin_edges:
#             hist.bucket_limit.append(edge)
#         for c in counts:
#             hist.bucket.append(c)
#
#         # Create and write Summary
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

# maintain all metrics required in this dictionary - these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}