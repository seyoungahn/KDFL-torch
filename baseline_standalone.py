import logging
import os
import numpy as np
import torch
from tqdm import tqdm
import utils
import models.resnet as resnet

def train(model, optimizer, criterion, trainloader, metrics, params):
    model.train()

    summaries = []
    avg_loss = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(trainloader)) as t:
        for batch_idx, (x_train, y_train) in enumerate(trainloader):
            # Move to GPU if available
            if params.cuda:
                x_train, y_train = x_train.cuda(non_blocking=True), y_train.cuda(non_blocking=True)

            x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation (train acc, train loss)
            if batch_idx % params.save_summary_steps == 0:
                y_pred = y_pred.data.cpu().numpy()
                y_train = y_train.data.cpu().numpy()

                summary = {metric:metrics[metric](y_pred, y_train) for metric in metrics}
                summary['loss'] = loss.item()
                summaries.append(summary)

            avg_loss.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
            t.update()
        print(summaries)
        metrics_mean = {metric:np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)
        return metrics_mean

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
    for batch_idx, (x_valid, y_valid) in enumerate(validloader):
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
    # print("Evaluation summaries")
    # print(summaries)
    metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean

def train_and_evaluate(model, trainloader, validloader, optimizer, criterion, metrics, params, model_dir, restore_file=None):
    ## Train the model and evaluate every epoch.
    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_valid_acc = 0

    if params.model_version == "resnet18":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "cnn":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

    for epoch in range(params.num_epochs):

        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))

        train(model, optimizer, criterion, trainloader, metrics, params)

        scheduler.step()

        valid_metrics = evaluate(model, criterion, validloader, metrics, params)

        valid_acc = valid_metrics['accuracy']
        is_best = valid_acc >= best_valid_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch+1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best accuracy")
            best_valid_acc = valid_acc

            # Save best validation metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_valid_best_weights.json")
            utils.save_dict_to_json(valid_metrics, best_json_path)

        # Save latest valid metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_valid_last_weights.json")
        utils.save_dict_to_json(valid_metrics, last_json_path)

# if __name__ == '__main__':
#     # Load the parameters from json file
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_dir', default='experiments/baseline_standalone', help='Directory containing params.json')
#     parser.add_argument('--restore_file', default=None,
#                         help='Optional, name of the file in --model_dir containing weights to reload before training')  ## 'best' or 'train'
#     args = parser.parse_args()
#     json_path = os.path.join(args.model_dir, 'params.json')
#     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
#     params = utils.Params(json_path)
#
#     # use GPU if available
#     params.cuda = torch.cuda.is_available()
#
#     # Set the random seed for reproducible experiments
#     random.seed(230)
#     torch.manual_seed(230)
#     if params.cuda: torch.cuda.manual_seed(230)
#
#     # Set the logger
#     utils.set_logger(os.path.join(args.model_dir, 'train.log'))
#
#     # Create the input data pipeline
#     logging.info("Loading the datasets...")
#
#     # fetch dataloaders, considering full-set vs. sub-set scenarios
#     if params.subset_percent < 1.0:
#         trainloader = datautils.fetch_subset_dataloader('train', params)
#     else:
#         trainloader = datautils.fetch_dataloader('train', params)
#
#     testloader = datautils.fetch_dataloader('test', params)
#
#     logging.info("- done.")
#
#     model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
#     optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
#     # fetch loss function and metrics
#     loss_fn = utils.loss_function
#     metrics = utils.metrics
#
#     # Train the model
#     logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
#     train_and_evaluate(model, trainloader, testloader, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)