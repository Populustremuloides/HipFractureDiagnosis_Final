import sys
sys.path.append('../')
from ModelScripts.InverseFourierModel import *
import torch.optim as optim
import torchvision.models as models
import argparse
from torch.utils.data import DataLoader
from TrainingScripts.TrainModelLoop2 import *
from TrainingScripts.Sensitivity import *
from TrainingScripts.Specificity import *
import pickle
import time
import torchvision
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--modelType", help="pretrained model. Options include: alex, vgg19, resnet101, densenet121, inception, googlenet, wideResnet50")
parser.add_argument("-e", "--numEpochs", help="a single int: number of epochs to trian for.", action="store", required=True)
parser.add_argument("-b", "--batchSize", help="a single int: number of images to be included in each batch.", action="store", required=True)
parser.add_argument("-d", "--dataFile", help="the name of a file that contains on the first line the path of the folder that contains the images to be trained, and on the second line, the path of hte folder that contains the validation data. Both folders must in the pytorch ImageFolder format.", required=True)
parser.add_argument("-i", "--evalInterval", help="a single int: the number of batches to train on before evaluating the model. Default is 5.", action="store")
parser.add_argument("-o", "--outputFolder", help="a directory where the output from tests is saved.", action="store", required=True)
# **********************************************************


arguments = parser.parse_args()

modelType = str(arguments.modelType)
numEpochs = int(arguments.numEpochs)
batchSize = int(arguments.batchSize)

dataFile = str(arguments.dataFile)
evalInterval = int(arguments.evalInterval)
outputFolder = str(arguments.outputFolder)

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

if evalInterval is None:
    evalInterval = 5

with open(dataFile, "r+") as df:
    trainDataPath = df.readline()
    trainDataPath = trainDataPath.replace("\n","")
    valDataPath = df.readline()
    valDataPath = valDataPath.replace("\n","")
    print()
    print('trainData path: ' + str(trainDataPath))
    print('evaluationData path: ' + str(valDataPath))
    print()

trainDataset = torchvision.datasets.ImageFolder(trainDataPath, transform=torchvision.transforms.ToTensor())
trainLoader = DataLoader(trainDataset,batch_size=batchSize, shuffle=True)
numCategories = len(trainDataset.classes)
valDataset = torchvision.datasets.ImageFolder(valDataPath, transform=torchvision.transforms.ToTensor())
valLoader = DataLoader(valDataset,batch_size=batchSize, shuffle=True)

print(trainDataset.classes)
print(valDataset.classes)

print(trainDataset.class_to_idx)
#print(trainDataset.class_to_idx.keys())
#print(list(trainDataset.class_to_idx.keys()))
print(valDataset.class_to_idx)
#quit()
# ****************************************************

# models:
if modelType == "alex":
    model = models.alexnet(pretrained=True)
    print(model)
    numFeatures = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(numFeatures, numCategories)
elif modelType == "vgg19":
    model = models.vgg16(pretrained=True)
    print(model)
    numFeatures = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(numFeatures, numCategories)
elif modelType == "resnet152":
    model = models.resnet152(pretrained=True)
    print(model)
    numFeatures = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(numFeatures, numCategories)
    model.fc.requires_grad = True
elif modelType == "densenet201":
    model = models.densenet201(pretrained=True)
    numFeatures = model.classifier.in_features
    model.classifier = nn.Linear(numFeatures, numCategories)
    print(model)
elif modelType == "inception":
    model = models.inception_v3(pretrained=True)
    print(model)
elif modelType == "googlenet":
    model = models.googlenet(pretrained=True)
    numFeatures = model.fc.in_features
    model.fc = nn.Linear(numFeatures, numCategories)
    print(model)
elif modelType == "wideResNet101":
    model = models.wide_resnet101_2(pretrained=True)
    numFeatures = model.fc.in_features
    model.fc = nn.Linear(numFeatures, numCategories)
    print(model)
else:
    print("unrecognized model type: " + modelType)
    quit()


model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

lossFunction = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
sensitivityFunction = GetSensitivities()
specificityFunction = GetSpecificities()

startTime = time.time()

lossList, sensitivityList, specificityList, valLossList, valSensitivities, valSpecificities, accList, matrixList = trainModelLoop(model, 
        optimizer, trainLoader, valLoader, lossFunction, sensitivityFunction, 
        specificityFunction, numEpochs, evalInterval, numCategories, cuda=True, labelSmooth=True)

endTime = time.time()
print()
print("total time: " + str(endTime - startTime))

labelCategories = ["LF","L-NF","RF","R-NF"] #list(trainDataset.class_to_idx.keys())
predCategories = ["LF","L-NF","RF","R-NF"]


dataDict = {}
matrix = matrixList[-1][1]
for i in range(matrix.shape[0]):
    print(matrix[i])
    print(labelCategories[i])
    dataDict[labelCategories[i]] = matrix[i]
df = pd.DataFrame.from_dict(dataDict)
df.index = predCategories
print(df)


print(matrixList[-1][1])
#plt.show()


suffix =  modelType + "_mediumImages_yesLabelSmoothing_numEpochs:" + str(numEpochs) + "_batchSize" + str(batchSize) + ".p"
lossName = outputFolder + "/train_loss_" + suffix 
valLossName = outputFolder + "/val_loss_" + suffix
valSensitivityName = outputFolder + "/val_sensitivty_" + suffix 
valSpecificityName = outputFolder + "/val_specificity_" + suffix 


vLossIndex, vLoss = zip(*valLossList)
plt.plot(lossList, label="train loss")
plt.plot(vLossIndex, vLoss, label="validaiton loss")
plt.xlabel("batch")
plt.ylabel("loss")
plt.legend()
figureName = modelType + "_mediumImages_yesLabelSmoothing_numEpochs:" + str(numEpochs) + "_batchSize_" + str(batchSize)
plt.savefig(outputFolder + "/" + figureName)
plt.clf()


sns.heatmap(df, annot=True)
plt.title("Confusion Matrix - GoogleNet Pretrained on HF Radiographs")
plt.xlabel("correct label")
plt.ylabel("model prediction")
plt.savefig(outputFolder + "/confusionMatrix" + figureName + ".png")
plt.clf()

pickle.dump(lossList, open(lossName, "wb"))
pickle.dump(valLossList, open(valLossName, "wb"))
pickle.dump(valSensitivities, open(valSensitivityName, "wb"))
pickle.dump(valSpecificities, open(valSpecificityName, "wb"))

modelName = lossName.replace("train_loss-", "model-")
torch.save(model.state_dict(), modelName)

summaryName = lossName.replace("train_loss-","summary-")
summaryName = summaryName + ".txt"

with open(summaryName, "w+") as summary:
    
    summary.write("totalTime: " + str(endTime - startTime) + "\n\n")
    summary.write("training data used: " + str(trainDataPath) + "\n")
    summary.write("validation data used: " + str(valDataPath) + "\n\n")
    summary.write("num epochs: " + str(numEpochs) + "\n")
    summary.write("batch size: " + str(batchSize) + "\n")
    summary.write("final validation accuracy: " + str(accList[-1]) + "\n")
    summary.write("final validation loss: " + str(valLossList[-1]) + "\n")

    summary.write(str(trainDataset.classes) + "\n")
     
    summary.write("final validation sensitvities:\n")
    for category in valSensitivities.keys():
        sensitivity = valSensitivities[category][-1]
        summary.write(str(category) + ": " + str(sensitivity) + "\n")
    summary.write("\n")
    summary.write("final validation specificities\n")
    for category in valSpecificities.keys():
        specificity = valSpecificities[category][-1]
        summary.write(str(category) + ": " + str(specificity) + "\n")


