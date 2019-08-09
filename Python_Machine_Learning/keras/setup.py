# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from IPython.display import HTML

# Common imports
import numpy as np
import os
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import matplotlib.image as mpimg
def plot_external(img):
    img_name = os.path.join(".", "images",img)
    plots = mpimg.imread(img_name)
    plt.axis("off")
    plt.imshow(plots)    
    plt.show()
    
def plot_external2(imgpath):    
    plots = mpimg.imread(imgpath)
    plt.axis("off")
    plt.imshow(plots)    
    plt.show()  

# batch function
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        
from tensorflow.keras.backend import clear_session

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("directory created!!")
    else:
        print("directory exist!!")
        

class LossAndError(tf.keras.callbacks.Callback):
    def __init__(self, epochs, class_mode, validation=True):
        self.epochs = epochs
        self.validation = validation
        
        # class_mode: binary_accuracy, sparse_categorical_crossentropy, 
        # mean_absolute_error, mean_squared_error, categorical_crossentropy        
        self.class_mode = class_mode
        self.val_class_mode = "val_" + self.class_mode
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch                
              
    def on_train_batch_end(self, batch, logs=None):        
        print('\rEpoch {}/{}, loss: {:.5f}, {}: {:.5f}'.format(self.epoch+1, self.epochs,
                                                            logs['loss'], 
                                                            self.class_mode, 
                                                            logs[self.class_mode]), end="")
              
    def on_epoch_end(self, epoch, logs=None):        
        if epoch == 0:
            print("\nhistory key: {}".format(list(logs.keys())))
            
        if self.epochs >= 5: 
            if self.epochs // 5 == 1:
                self.n_batch_steps = 2
            else:
                self.n_batch_steps = self.epochs // 5
            
            if (epoch+1) % self.n_batch_steps == 0 or epoch+1 == self.epochs:
                if self.validation:
                    print("\rEpoch {}/{}".format(epoch+1, self.epochs), end="")
                    print(", loss: {:.5f}".format(logs["loss"]), end="")
                    print(", {}: {:.5f}".format(self.class_mode, logs[self.class_mode]), end="")
                    print(", val_loss: {:.5f}".format(logs["val_loss"]), end="")
                    print(", {}: {:.5f}".format(self.val_class_mode, logs[self.val_class_mode]))
                else:
                    print("\rEpoch {}/{}".format(epoch+1, self.epochs), end="")
                    print(", loss: {:.5f}".format(logs["loss"]), end="")
                    print(", {}: {:.5f}".format(self.class_mode, logs[self.class_mode]))
        else:            
            self.n_batch_steps = self.epochs            

            if self.validation:
                print("\rEpoch {}/{}".format(epoch+1, self.epochs), end="")
                print(", loss: {:.5f}".format(logs["loss"]), end="")
                print(", {}: {:.5f}".format(self.class_mode, logs[self.class_mode]), end="")
                print(", val_loss: {:.5f}".format(logs["val_loss"]), end="")
                print(", {}: {:.5f}".format(self.val_class_mode, logs[self.val_class_mode]))
            else:
                print("\rEpoch {}/{}".format(epoch+1, self.epochs), end="")
                print(", loss: {:.5f}".format(logs["loss"]), end="")
                print(", {}: {:.5f}".format(self.class_mode, logs[self.class_mode]))
        

            
def loss_and_acc_plot(history, class_mode, validation=True):        
    if validation:
        history_dict = history.history
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        acc = history_dict[class_mode]
        val_class_mode = "val_" + class_mode
        val_acc = history_dict[val_class_mode]
        epochs = range(1, len(loss)+1)

        plt.figure(figsize=(20, 8))
        plt.subplot(121)
        plt.plot(epochs, loss, "bo-", label="training loss")
        plt.plot(epochs, val_loss, "ro--", markerfacecolor="red", markeredgecolor="red", label="validation loss")
        plt.title("Training and validation loss", fontsize=18)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(122)
        plt.plot(epochs, acc, "bo-", label="training accuracy")
        plt.plot(epochs, val_acc, "ro--", markerfacecolor="red", markeredgecolor="red", label="validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation accuracy", fontsize=18)
        plt.legend()
        plt.show()
    
    else:
        history_dict = history.history
        loss = history_dict["loss"]
        acc = history_dict[class_mode]
        epochs = range(1, len(loss)+1)
        
        plt.figure(figsize=(20, 8))
        plt.subplot(121)
        plt.title("Training loss", fontsize=18)
        plt.plot(epochs, loss, "bo-", label="training loss")
        
        plt.subplot(122)
        plt.plot(epochs, acc, "bo-", label="training acc")
        plt.title("Training accuracy", fontsize=18)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()    