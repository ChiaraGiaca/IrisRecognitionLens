import torch
import torch.nn as nn
import time
import copy
from tqdm import tqdm
from sklearn import metrics
from train_config import TrainConfig, create_train_config
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import torch.nn.functional as F
import random
import pandas as pd
from IPython.display import display
import csv
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/content/drive/MyDrive/Biometric Systems/iris-recognition')
from utils import file_utils

sns.reset_orig()


CHECKPOINT_FILE_NAME = "iris_recognition_trained_model-2.pt"

def train_model(train_config: TrainConfig,
                model: nn.Module,
                criterion,
                optimizer,
                num_epochs):
    print("\nTraining with dataset: ", train_config.dataset)
    print(train_config.class_names)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    accuracies = {
        "train": [],
        "val": []
    }
    
    losses = {
      "train":[],
      "val":[]
    }

    acc = {
      "train":[],
      "val":[]
    }



    torch.cuda.empty_cache()
    mapping_dataset = {'train': {}, 'val': {}}
    
    
    class_indexes = train_config.data['val'].class_to_idx
    confusion_matrix_pred = []
    confusion_matrix_true = []
    best_epoch = -1
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Separate logs
            print()

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            bar_desc = f"EPOCH {epoch + 1}/{num_epochs}, {phase.upper()}"
            counter=0
            for inputs, labels, paths in tqdm(train_config.loaders[phase],
                                       desc=bar_desc):
      
                inputs = inputs.to(train_config.device)
                labels = labels.to(train_config.device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                   
                    outputs = model(inputs)
                    probabilities = outputs[:, 1]
                    
                      
                    _, preds = torch.max(outputs, 1)
                    #loss = criterion(anchor, pos, neg)
                    
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
              
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                counter+=1


            epoch_loss = running_loss / train_config.data_sizes[phase]
            epoch_acc = running_corrects.double() / train_config.data_sizes[
                phase]
         
            accuracies[phase].append(epoch_acc.cpu().numpy())
            acc[phase].append(epoch_acc.cpu().item())
            losses[phase].append(epoch_loss)
            
            print(f"Loss: {epoch_loss:.3}, "
                  f"{phase.lower()} accuracy: {epoch_acc:.2%}")

            
            # deep copy the model
            early_stop_thresh = 5
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:}m "
          f"{time_elapsed % 60:.2}s")
    print(f"Best validation accuracy: {best_acc:.4}")
    # load best model weights
    model.load_state_dict(best_model_wts)
    print()
    plot(acc,losses)
    print()

    return model, accuracies,acc, losses


def plot(accuracies,losses):
  
  #plot of accuracies
  plt.plot(accuracies["train"])
  plt.plot(accuracies["val"])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

  print()

  #plot of losses
  plt.plot(losses["train"])
  plt.plot(losses["val"])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  return

  
def run():
    train_config = create_train_config()
    trained_model, accuracies, acc, losses = train_model(
        train_config=train_config,
        model=train_config.model,
        optimizer=train_config.optimizer,
        criterion=train_config.criterion,
        num_epochs=train_config.num_epochs
    )

    #saving model parameters in a dictionary to use it later
    print(train_config.class_names)
    checkpoint_dict = {
        "model_state_dict": trained_model.state_dict(),
        "optimizer_state_dict": train_config.optimizer.state_dict(),
        "train_accuracies": accuracies["train"],
        "validation_accuracies": accuracies["val"],
        "classes": train_config.class_names
    }

    torch.save(checkpoint_dict, CHECKPOINT_FILE_NAME)

run()
