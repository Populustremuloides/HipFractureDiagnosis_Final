import sys
sys.path.append('../')
from TrainingScripts.SmoothLabel import *
import torch
import numpy as np
import gc

def getMeanTupleValues(tupleList, numCategories):
    
    meanTuple = ()
    for i in range(numCategories): # for each category
        categoryValues = []
        for value in tupleList: # find all values
            categoryValues.append(value[i])

        categoryMean = np.mean(categoryValues) # calculate the mean
        meanTuple = meanTuple + (categoryMean,) # add it to the k-tuple
    
    return meanTuple

def evaluateModel(model, valLoader, lossFunction, sensitivityFunction, specificityFunction, cuda, numCategories, labelSmoothing=False):

    print(valLoader.dataset.class_to_idx)

    model.eval()

    lossList = []
    accuracyList = []

    sensitivityList = []
    specificityList = []
    
    predictions = torch.LongTensor([]).cuda()
    ys = torch.LongTensor([]).cuda()
    for batch, (x,y) in enumerate(valLoader):
        if cuda:
            y = y.cuda()
            x = x.cuda()

        yHat = model(x)

        predictedCategory = torch.argmax(yHat, dim=1)
        predictedCategory = (predictedCategory * -1) + numCategories - 1 # hack because the model's predictions seem to be reversed somehow 


        predictions = torch.cat((predictions, predictedCategory), dim=0)
        ys = torch.cat((ys, y), dim=0)

        if labelSmoothing:
            # note that if labelSmoothing = True and the loss is something like BCE, it will throw an error
            smoothY = smoothLabel(y, numCategories, alpha = 0.2, cuda=cuda)
        else:
            smoothY = y

        loss = lossFunction(yHat, smoothY)
        lossList.append(loss.item())

    model.train()
    gc.collect()

    sensitivities = sensitivityFunction(predictions, ys, numCategories)
    specificities = specificityFunction(predictions, ys, numCategories)

    accuracy = torch.sum(predictions == ys).item() / predictions.shape[0]
    
    lineup = torch.zeros((numCategories, numCategories))
    for i in range(ys.shape[0]):
        x = ys[i]
        y = predictions[i]
        lineup[x][y] += 1

    #print("real values")
    #print(ys)
    #print("predictions")
    #print(predictions)
    lineup = lineup.cpu().detach().numpy()

    #print(lineup)
    #quit()

    print(sensitivities)
    print(specificities)
    print(accuracy)

    return np.mean(lossList), sensitivities, specificities, accuracy, lineup


