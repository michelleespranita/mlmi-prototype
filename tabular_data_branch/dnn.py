import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from keras.models import load_model, Model

import pandas as pd

def get_pretrained_keras_model():
    return load_model("./CF_Mortality.model")

def get_pretrained_weights_from_keras_model():
    keras_dnn = load_model("./CF_Mortality.model")
    keras_dnn_weights = keras_dnn.get_weights()
    return keras_dnn_weights

## Reconstruct Keras model in PyTorch

class DNN(nn.Module):
    def __init__(self, input_shape):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 48)
        self.fc5 = nn.Linear(48, 16)
#         self.output = nn.Linear(16, 2)
        
        self.dropout_50 = nn.Dropout(0.5)
        self.dropout_20 = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout_50(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_20(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_20(x)
        x = F.relu(self.fc4(x))
        x = self.dropout_20(x)
#         x = F.relu(self.fc5(x))
        return self.fc5(x)
#         x = self.dropout_50(x)
#         return F.softmax(self.output(x))


def get_tabular_keras_dnn():
    keras_pretrained_model = get_pretrained_keras_model()
    return Model(inputs=keras_pretrained_model.input, outputs=keras_pretrained_model.layers[-2].output)

## Transfer pre-trained weights from Keras model to PyTorch

def get_tabular_pytorch_dnn():
    keras_dnn_weights = get_pretrained_weights_from_keras_model()

    pytorch_dnn = DNN(128)
    pytorch_dnn.fc1.weight.data = torch.from_numpy(np.transpose(keras_dnn_weights[0]))
    pytorch_dnn.fc1.bias.data = torch.from_numpy(keras_dnn_weights[1])
    pytorch_dnn.fc2.weight.data = torch.from_numpy(np.transpose(keras_dnn_weights[2]))
    pytorch_dnn.fc2.bias.data = torch.from_numpy(keras_dnn_weights[3])
    pytorch_dnn.fc3.weight.data = torch.from_numpy(np.transpose(keras_dnn_weights[4]))
    pytorch_dnn.fc3.bias.data = torch.from_numpy(keras_dnn_weights[5])
    pytorch_dnn.fc4.weight.data = torch.from_numpy(np.transpose(keras_dnn_weights[6]))
    pytorch_dnn.fc4.bias.data = torch.from_numpy(keras_dnn_weights[7])
    pytorch_dnn.fc5.weight.data = torch.from_numpy(np.transpose(keras_dnn_weights[8]))
    pytorch_dnn.fc5.bias.data = torch.from_numpy(keras_dnn_weights[9])
    # pytorch_dnn.output.weight.data = torch.from_numpy(np.transpose(keras_dnn_weights[10]))
    # pytorch_dnn.output.bias.data = torch.from_numpy(keras_dnn_weights[11])

    return pytorch_dnn