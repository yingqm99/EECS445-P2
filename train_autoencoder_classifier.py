"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Train Autoencoder
    Trains an autoencoder to learn a sparse representation of images data
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_autoencoder.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.autoencoder import Autoencoder, AutoencoderClassifier
from train_common import *
from utils import config
import utils

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """

    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # forward + backward + optimize
        

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def evaluate_image():
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('cnn.num_classes'))
    # dataset = get_data_by_label(va_loader)
    # Model
    ae_classifier = AutoencoderClassifier(config('autoencoder.ae_repr_dim'),
        config('autoencoder.classifier.num_classes'))

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ae_classifier.parameters(),
        lr=config('autoencoder.classifier.learning_rate'))
    # print('Number of float-valued parameters:', count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    num_classes = 5
    performance = np.zeros(num_classes)
    for epoch in range(10):
        _train_epoch(tr_loader, ae_classifier, criterion, optimizer)
    total = [0,0,0,0,0]
    correct = [0,0,0,0,0]
    for X, y in va_loader:
        print(y.size())
        with torch.no_grad():
            output = ae_classifier(X)
            predicted = predictions(output.data)
            for c in range(y.size(0)):
                for i in range(5):
                    if y[c] == i:
                        total[i] = total[i] + 1
                        if predicted[c] == y[c]:
                            correct[i] = correct[i] + 1
    for i in range(5):
        print(correct[i]/total[i])

def main():
    evaluate_image()
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('autoencoder.classifier.num_classes'))

    ae_classifier = AutoencoderClassifier(config('autoencoder.ae_repr_dim'),
        config('autoencoder.classifier.num_classes'))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ae_classifier.parameters(),
        lr=config('autoencoder.classifier.learning_rate'))

    # freeze the weights of the encoder
    for name, param in ae_classifier.named_parameters():
        if 'fc1.' in name or 'fc2.' in name:
            param.requires_grad = False

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    ae_classifier, _, _ = restore_checkpoint(ae_classifier,
        config('autoencoder.checkpoint'), force=True, pretrain=True)
    print('Loading autoencoder classifier...')
    ae_classifier, start_epoch, stats = restore_checkpoint(ae_classifier,
        config('autoencoder.classifier.checkpoint'))

    fig, axes = utils.make_cnn_training_plot(name='Autoencoder Classifier')

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
        start_epoch, stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('autoencoder.classifier.num_epochs')):
        # Train model
        _train_epoch(tr_loader, ae_classifier, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
            epoch+1, stats)

        # Save model parameters
        save_checkpoint(ae_classifier, epoch+1,
            config('autoencoder.classifier.checkpoint'), stats)

    print('Finished Training')

    # Keep plot open
    utils.save_cnn_training_plot(fig, name='ae_clf')
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
