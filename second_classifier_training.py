# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:53:45 2019

Train the first vesicle classifier and evaluate its performance
on the training and validation datasets.

@author: imbroscb
"""

import os
import torch
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from CNNs_GaussianNoiseAdder import MultiClassPost, GaussianNoiseAddition

# set deterministic = True and look for cuda
torch.manual_seed(2809)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform and load the training data
transform_train = transforms.Compose(
    [transforms.Resize((80, 80)),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.RandomRotation(10),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomApply([GaussianNoiseAddition(0, 0.1)], p=0.1)])
training_dataset = datasets.ImageFolder(root=os.path.join(
    'data_post', 'train'), transform=transform_train)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset,
                                              batch_size=100, shuffle=True)

# transform and load the validation data
transform_test = transforms.Compose(
    [transforms.Resize((80, 80)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testing_dataset = datasets.ImageFolder(root=os.path.join(
    'data_post', 'test'), transform=transform_test)
validation_loader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                                batch_size=100, shuffle=False)

# check dimentions
print('Batch size: ', training_loader.batch_size)
print('Length training dataset: ', len(training_dataset))
print('Length validation dataset: ', len(testing_dataset))

# this can be used to check the loaded images and labels
dataiter = iter(training_loader)
images, labels = dataiter.next()
print('Image shape: ', images.shape)

# define the model, the loss and the optimizer
model = MultiClassPost(out=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

# inizialize some variables
loss_history = []
accuracy_history = []
precision_history = []
recall_history = []
val_loss_history = []
val_accuracy_history = []
val_precision_history = []
val_recall_history = []
images_counter_list = []
epochs = 55

# loop through the epochs
for e in range(epochs):

    # reset some variables at each epoch
    running_loss = 0.0
    running_correct = 0.0
    val_running_loss = 0.0
    val_running_correct = 0.0
    images_counter = 0
    val_images_counter = 0
    true_pos = 0
    val_true_pos = 0
    prec_denom = 0
    val_prec_denom = 0
    rec_denom = 0
    val_rec_denom = 0

    # loop through the training_loader
    for inputs, labels in training_loader:

        # sum up the number of images in the training dataset
        images_counter += inputs.shape[0]

        # set the inputs in the right shape and feed the classifier
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs[:, 0, :, :]
        inputs = inputs.view(inputs.shape[0], 1, inputs.shape[1],
                             inputs.shape[2])
        output = model.forward(inputs)

        # get the loss
        loss = criterion(output, labels)

        # set the gradients to zero
        optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the weights
        optimizer.step()

        # sum up the loss, the correct predictions and the true positives
        # for the training dataset for the current epoch
        running_loss += loss.item()
        _, preds = torch.max(output, 1)
        running_correct += torch.sum(preds == labels.data)
        for i in range(preds.shape[0]):
            if preds[i] == 1:
                if preds[i] == labels[i].data:
                    true_pos += 1

        # get the denominators for calculating precision and recall
        # for the training dataset for the current epoch
        if device == 'cpu':
            prec_denom += torch.sum(preds == 1.0)
            rec_denom += torch.sum(labels.data == 1.0)
        else:
            prec_denom += torch.sum(preds.cpu() == 1.0)
            rec_denom += torch.sum(labels.data.cpu() == 1.0)

    # now the validation dataset, no gradients calculation necessary
    else:
        with torch.no_grad():

            # loop through the validation_loader
            for val_inputs, val_labels in validation_loader:

                # sum up the number of images in the validation dataset
                val_images_counter += val_inputs.shape[0]

                # set the inputs in the right shape and feed the classifier
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_inputs = val_inputs[:, 0, :, :]
                val_inputs = val_inputs.view(val_inputs.shape[0],
                                             1, val_inputs.shape[1],
                                             val_inputs.shape[2])
                val_output = model.forward(val_inputs)

                # get the validation loss
                val_loss = criterion(val_output, val_labels)

                # sum up the loss, the correct predictions and the true
                # positives for the validation dataset for the current epoch
                val_running_loss += val_loss.item()
                _, val_preds = torch.max(val_output, 1)
                val_running_correct += torch.sum(val_preds == val_labels.data)
                for i in range(val_preds.shape[0]):
                    if val_preds[i] == 1:
                        if val_preds[i] == val_labels[i].data:
                            val_true_pos += 1

                # get the denominators for calculating precision and recall
                # for the validation dataset for the current epoch
                if device == 'cpu':
                    val_prec_denom += torch.sum(val_preds == 1.0)
                    val_rec_denom += torch.sum(val_labels.data == 1.0)
                else:
                    val_prec_denom += torch.sum(val_preds.cpu() == 1.0)
                    val_rec_denom += torch.sum(val_labels.data.cpu() == 1.0)

    # get loss, accuracy, precision and recall for the current epoch
    # for the training dataset...
    epoch_loss = running_loss/images_counter*100
    epoch_accuracy = running_correct/images_counter*100
    true_pos = np.array(true_pos)
    prec_denom = prec_denom.numpy()
    rec_denom = rec_denom.numpy()
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)
    precision_history.append(true_pos/prec_denom*100)
    recall_history.append(true_pos/rec_denom*100)
    # and for the validation dataset
    val_epoch_loss = val_running_loss/val_images_counter*100
    val_epoch_accuracy = val_running_correct/val_images_counter*100
    val_true_pos = np.array(val_true_pos)
    val_prec_denom = val_prec_denom.numpy()
    val_rec_denom = val_rec_denom.numpy()
    val_loss_history.append(val_epoch_loss)
    val_accuracy_history.append(val_epoch_accuracy)
    val_precision_history.append(val_true_pos/val_prec_denom*100)
    val_recall_history.append(val_true_pos/val_rec_denom*100)

# print the results
    print('Epoch, training loss, training accuracy: {:},{:.4f},{:.4f}'.
          format(e, epoch_loss, epoch_accuracy))
    print('Epoch, validation loss, validation accuracy: {:},{:.4f},{:.4f}'.
          format(e, val_epoch_loss, val_epoch_accuracy))
    print('Epoch, training precision, training recall: {:},{:.4f},{:.4f}'.
          format(e, precision_history[-1], recall_history[-1]))
    print('Epoch, validation precision, validation recall: {:},{:.4f},{:.4f}'.
          format(e, val_precision_history[-1],
                 val_recall_history[-1]))
    print('------------------------------------------------------------------')

    # here change accordingly the directory where to save the model weights
    torch.save(model.state_dict(), os.path.join('/path', 'model_dir',
                                                'epoch-{}.pth'.format(e)))

# get the f1_score for the training dataset
f1_train = []
for i in range(epochs):
    f1_train.append(2*(precision_history[i]*recall_history[i])
                    / (precision_history[i]+recall_history[i]))

# get the f1_score for the validation dataset
f1_val = []
for i in range(epochs):
    f1_val.append(2*(
        val_precision_history[i]*val_recall_history[i]) /
        (val_precision_history[i]+val_recall_history[i]))
