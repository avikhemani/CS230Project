from colorama import Fore, Back, Style
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier
from torch import nn, optim, tensor, from_numpy, FloatTensor, device, cuda, max as torchmax
import torch.nn.functional as F

# Paths to data
TRAIN_PATH = "./AudioMNIST/data/train"
DEV_PATH = "./AudioMNIST/data/dev"
TEST_PATH = "./AudioMNIST/data/test"

# Constants
AUDIO_LEN = 8000
NUM_EPOCHS = 25
BATCH_SIZE = 64

# GPU availability
torchDevice = device('cuda' if cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(AUDIO_LEN, 2000)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.output = nn.Linear(2000, 10)
        nn.init.xavier_uniform_(self.output.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x

def getInputOutputData(directory):
    fileNames = os.listdir(directory)
    n = len(fileNames)
    xTrain = []
    yTrain = []
    for (i, fileName) in enumerate(fileNames):
        if i == 15000: break
        if (i+1) % 500 == 0: print(Fore.GREEN + "Loaded " + str(i+1) + " out of " + str(n))
        timeSeries, samplingRate = librosa.load(os.path.join(directory, fileName), sr=AUDIO_LEN)
        diff = AUDIO_LEN - timeSeries.shape[0]
        input = np.append(timeSeries, np.asarray([0 for _ in range(diff)]))
        xTrain.append(input)

        digit = int(fileName[0])
        output = [0 for _ in range(10)]
        output[digit] = 1
        yTrain.append(np.asarray(output))

    print(Style.RESET_ALL)
    return np.asarray(xTrain), np.asarray(yTrain)

def neuralNetworkTorch(xTrain, yTrain):
    length = len(yTrain)
    xTrainTensors, yTrainTensors = np.array_split(xTrain, length//BATCH_SIZE), np.array_split(yTrain, length//BATCH_SIZE)
    for i in range(len(yTrainTensors)):
        xTrainTensors[i] = from_numpy(xTrainTensors[i]).type(FloatTensor).to(torchDevice)
        yTrainTensors[i] = from_numpy(yTrainTensors[i]).to(torchDevice)
    nnet = NeuralNetwork().to(torchDevice)
    loss_function = nn.MultiLabelSoftMarginLoss()
    #loss_function = nn.MultiLabelMarginLoss()
    optimizer = optim.Adam(nnet.parameters(), lr=0.002)

    nnet.train()
    for epoch in range(NUM_EPOCHS):
        for xTrainTensor, yTrainTensor in zip(xTrainTensors, yTrainTensors):
            output = nnet(xTrainTensor) # forward propogation
            loss = loss_function(output, yTrainTensor) # loss calculation
            optimizer.zero_grad()
            loss.backward() # backward propagation
            optimizer.step() # weight optimization

        print("Epoch:", epoch+1, "Training Loss: ", loss.item())

    return nnet

def reportAccuracy(prediction, actual):
    print(Style.RESET_ALL)
    assert len(prediction) == len(actual)

    accurate, total = 0, len(prediction)
    for i in range(total):
        if prediction[i].argmax(0) == actual[i].argmax(0): accurate += 1

    print("Accuracy ", accurate/total)

def main():
    xTrain, yTrain = getInputOutputData(TRAIN_PATH)
    xTest, yTest = getInputOutputData(DEV_PATH)

    # ------ NeuralNetworkTorch -----
    nnet = neuralNetworkTorch(xTrain, yTrain)
    nnet.eval()
    xTestTensor = from_numpy(xTest).type(FloatTensor).to(torchDevice)
    output = nnet(xTestTensor)
    numpyOutput = output.detach().numpy()
    prediction = np.zeros_like(numpyOutput)
    prediction[np.arange(len(numpyOutput)), numpyOutput.argmax(1)] = 1
    print(prediction)
    reportAccuracy(prediction, yTest)

if __name__ == '__main__':
    main()
