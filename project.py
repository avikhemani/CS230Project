from colorama import Fore, Back, Style
import os
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier
from torch import nn, optim, tensor, from_numpy, FloatTensor, device, cuda, max as torchmax
import torch.nn.functional as F
import random

# Paths to data
TRAIN_PATH = "./data/train"
DEV_PATH = "./data/dev"
TEST_PATH = "./data/test"

# Constants
AUDIO_LEN = 16000
NUM_EPOCHS = 25
BATCH_SIZE = 64

# GPU availability
torchDevice = device('cuda' if cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(AUDIO_LEN, 1000)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.output = nn.Linear(1000, 10)
        nn.init.xavier_uniform_(self.output.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(3 * 3 * 10, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

def getInputOutputData(directory, createImage):
    fileNames = os.listdir(directory)
    n = len(fileNames)
    random.seed(1)
    random.shuffle(fileNames)

    xTrain = []
    yTrain = []
    pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
    pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))
    for (i, fileName) in enumerate(fileNames):
        if (i+1) % 500 == 0: print(Fore.GREEN + "Loaded " + str(i+1) + " out of " + str(n))
        timeSeries, samplingRate = librosa.load(os.path.join(directory, fileName), sr=AUDIO_LEN)
        #timeSeries = librosa.util.normalize(timeSeries)

        if not createImage:
            timeSeries = pad1d(timeSeries, AUDIO_LEN)
            xTrain.append(timeSeries)

            # plt.figure()
            # plt.plot(timeSeries)
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.plot()
            # plt.show()
        else:
            spectogram = np.abs(librosa.stft(timeSeries))
            spectogram = pad2d(spectogram, AUDIO_LEN//1000 * 2)
            spectogram = librosa.amplitude_to_db(spectogram)
            # ld.specshow(spectogram, y_axis='linear')
            # plt.show()
            xTrain.append(spectogram)

        digit = int(fileName[0])
        output = [0 for _ in range(10)]
        output[digit] = 1
        yTrain.append(np.asarray(output))

    print(Fore.GREEN + "Loading completed")
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
    optimizer = optim.Adam(nnet.parameters(), lr=0.001)

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

def convNetworkTorch(xTrain, yTrain):
    length = len(yTrain)
    xTrainTensors, yTrainTensors = np.array_split(np.expand_dims(xTrain, axis=1), length//BATCH_SIZE), np.array_split(yTrain, length//BATCH_SIZE)
    print(xTrainTensors[0].shape)
    for i in range(len(yTrainTensors)):
        xTrainTensors[i] = from_numpy(xTrainTensors[i]).type(FloatTensor).to(torchDevice)
        yTrainTensors[i] = from_numpy(yTrainTensors[i]).to(torchDevice)
    convnet = ConvNetwork().to(torchDevice)
    loss_function = nn.MultiLabelSoftMarginLoss()
    #loss_function = nn.MultiLabelMarginLoss()
    optimizer = optim.Adam(nnet.parameters(), lr=0.001)

    convnet.train()
    for epoch in range(NUM_EPOCHS):
        for xTrainTensor, yTrainTensor in zip(xTrainTensors, yTrainTensors):
            output = convnet(xTrainTensor) # forward propogation
            loss = loss_function(output, yTrainTensor) # loss calculation
            optimizer.zero_grad()
            loss.backward() # backward propagation
            optimizer.step() # weight optimization

        print("Epoch:", epoch+1, "Training Loss: ", loss.item())

    return convnet

def reportAccuracy(xTest, yTest):
    xTestTensor = from_numpy(xTest).type(FloatTensor).to(torchDevice)
    output = nnet(xTestTensor)
    numpyOutput = output.cpu().detach().numpy()
    prediction = np.zeros_like(numpyOutput)
    prediction[np.arange(len(numpyOutput)), numpyOutput.argmax(1)] = 1

    print(Style.RESET_ALL)
    assert len(prediction) == len(yTest)

    accurate, total = 0, len(prediction)
    for i in range(total):
        if prediction[i].argmax(0) == yTest[i].argmax(0): accurate += 1

    print("Accuracy ", accurate/total)

def main():
    useConvNet = True
    xTrain, yTrain = getInputOutputData(TRAIN_PATH, useConvNet)
    xDev, yDev = getInputOutputData(DEV_PATH, useConvNet)
    xTest, yTest = getInputOutputData(TEST_PATH, useConvNet)
    np.save("./xTrainConv", xTrain)
    np.save("./yTrainConv", yTrain)
    np.save("./xDevConv", xDev)
    np.save("./yDevConv", yDev)
    np.save("./xTestConv", xTest)
    np.save("./yTestConv", yTest)

    if not useConvNet:
        # ------ NeuralNetworkTorch -----
        nnet = neuralNetworkTorch(xTrain, yTrain)
        nnet.eval()
        xTestTensor = from_numpy(xTest).type(FloatTensor).to(torchDevice)
        output = nnet(xTestTensor)
        numpyOutput = output.cpu().detach().numpy()
        prediction = np.zeros_like(numpyOutput)
        prediction[np.arange(len(numpyOutput)), numpyOutput.argmax(1)] = 1
        reportAccuracy(prediction, yTest)
    else:
        # ------ ConvNetworkTorch -----
        convnet = convNetworkTorch(xTrain, yTrain)
        convnet.eval()
        xTestTensor = from_numpy(np.expand_dims(xTest, axis=1)).type(FloatTensor).to(torchDevice)
        output = convnet(xTestTensor)
        numpyOutput = output.cpu().detach().numpy()
        prediction = np.zeros_like(numpyOutput)
        prediction[np.arange(len(numpyOutput)), numpyOutput.argmax(1)] = 1
        reportAccuracy(prediction, yTest)

if __name__ == '__main__':
    main()
