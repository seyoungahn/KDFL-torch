import os
import utils
import torch
import random
import logging
import datautils
import models.resnet as resnet
import models.alexnet as alexnet
import torch.optim as optim
import baseline_KD
import baseline_standalone
import csv

class Experiments:
    def __init__(self, exp_name, params):
        self.model_dir = os.path.join("experiments", exp_name)
        if not os.path.isdir(self.model_dir):
            utils.mkdir_p(self.model_dir)
        self.params = params

        # use GPU if available
        self.params.cuda = torch.cuda.is_available()

        # Set the random seed for reproducible experiments
        random.seed(233)
        torch.manual_seed(233)
        if self.params.cuda: torch.cuda.manual_seed(233)

    def set_baseline_dataset(self):
        utils.set_logger(os.path.join(self.model_dir, 'train.log'))
        logging.info("Loading the datasets...")

        if self.params.subset_percent < 1.0:
            trainloader = datautils.fetch_subset_dataloader('train', self.params)
        else:
            trainloader = datautils.fetch_dataloader('train', self.params)

        testloader = datautils.fetch_dataloader('test', self.params)

        logging.info("- done.")
        self.trainloader = trainloader
        self.testloader = testloader

    def set_federated_dataset(self):
        #TODO: participant별로 데이터를 나누어 생성, IID, non-IID case 모두 cover해야함
        pass

    def train_baseline_standalone(self):
        if self.params.model_version == 'resnet18':
            model = resnet.ResNet18().cuda() if self.params.cuda else resnet.ResNet18()
        elif self.params.model_version == 'alexnet':
            model = alexnet.AlexNet().cuda() if self.params.cuda else alexnet.AlexNet()
        optimizer = optim.SGD(model.parameters(), lr=self.params.learning_rate, momentum=0.9, weight_decay=5e-4)
        loss_function = utils.loss_function
        metrics = utils.metrics

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(self.params.num_epochs))

        best_valid_acc = 0

        if self.params.model_version == "resnet18":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        elif self.params.model_version == "alexnet":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

        for epoch in range(self.params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, self.params.num_epochs))

            train_metrics = baseline_standalone.train(model, optimizer, loss_function, self.trainloader, metrics, self.params)

            scheduler.step()

            valid_metrics = baseline_standalone.evaluate(model, loss_function, self.testloader, metrics, self.params)

            valid_acc = valid_metrics['accuracy']
            is_best = valid_acc >= best_valid_acc

            # Record experiment results
            with open(self.model_dir + "/result.csv", 'a') as f:
                writer = csv.writer(f)
                row = [train_metrics['loss'], train_metrics['accuracy'], valid_metrics['loss'], valid_metrics['accuracy']]
                writer.writerow(row)

            # Save weights
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            }, is_best=is_best, checkpoint=self.model_dir)

            if is_best:
                logging.info("- Found new best accuracy")
                best_valid_acc = valid_acc

                # Save best validation metrics in a json file in the model directory
                best_json_path = os.path.join(self.model_dir, "metrics_valid_best_weights.json")
                utils.save_dict_to_json(valid_metrics, best_json_path)

            # Save latest valid metrics in a json file in the model directory
            last_json_path = os.path.join(self.model_dir, "metrics_valid_last_weights.json")
            utils.save_dict_to_json(valid_metrics, last_json_path)

    def train_baseline_KD(self):
        if self.params.model_version == 'resnet18':
            model = resnet.ResNet18().cuda() if self.params.cuda else resnet.ResNet18()
        elif self.params.model_version == 'alexnet':
            model = alexnet.AlexNet().cuda() if self.params.cuda else alexnet.AlexNet()
        optimizer = optim.SGD(model.parameters(), lr=self.params.learning_rate, momentum=0.9, weight_decay=5e-4)
        loss_function_KD = utils.loss_function_kd
        metrics = utils.metrics

        teacher_model = resnet.ResNet18()
        teacher_checkpoint = 'experiments/baseline_standalone_resnet18/best.pth.tar'
        teacher_model = teacher_model.cuda() if self.params.cuda else teacher_model
        utils.load_checkpoint(teacher_checkpoint, teacher_model)

        # Train the model with KD
        logging.info("Experiment - model version: {}".format(self.params.model_version))
        logging.info("Starting training for {} epoch(s)".format(self.params.num_epochs))
        logging.info("First, loading the teacher model and computing its outputs...")


        best_valid_acc = 0.0

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        for epoch in range(self.params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, self.params.num_epochs))

            train_metrics = baseline_KD.train_kd(model, teacher_model, optimizer, loss_function_KD, self.trainloader, metrics, self.params)

            scheduler.step()

            valid_metrics = baseline_KD.evaluate_kd(model, teacher_model, loss_function_KD, self.testloader, metrics, self.params)

            valid_acc = valid_metrics['accuracy']
            is_best = valid_acc >= best_valid_acc

            # Record experiment results
            with open(self.model_dir + "/result.csv", 'a') as f:
                writer = csv.writer(f)
                row = [train_metrics['loss'], train_metrics['accuracy'], valid_metrics['loss'],
                       valid_metrics['accuracy']]
                writer.writerow(row)

            # Save weights
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            }, is_best=is_best, checkpoint=self.model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_valid_acc = valid_acc

                # Save best valid metrics in a JSON file in the model directory
                best_json_path = os.path.join(self.model_dir, "metrics_valid_best_weights.json")
                utils.save_dict_to_json(valid_metrics, best_json_path)

            # Save latest valid metrics in a JSON file in the model directory
            last_json_path = os.path.join(self.model_dir, "metrics_valid_last_weights.json")
            utils.save_dict_to_json(valid_metrics, last_json_path)


    def train_baseline_FL(self):
        pass