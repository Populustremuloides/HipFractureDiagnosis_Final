import sys
sys.path.append('../')
from TrainingScripts.EvaluateModel2 import *
from TrainingScripts.SmoothLabel import *
from tqdm import tqdm
import gc

def trainModelLoop(model, optimizer, trainLoader, valLoader, lossFunction, sensitivityFunction, specificityFunction, numEpochs, evaluationInterval, numCategories, cuda=True, labelSmooth=False):
    model.train()
    
    numSteps = 10

    stepInterval = int(numEpochs // numSteps)
    print(stepInterval)

    numStepsTaken = 0

    lossList = []

    sensitivitiesDict = {}
    specificitiesDict = {}

    for category in range(numCategories):
        sensitivitiesDict["category_" + str(category)] = []
        specificitiesDict["category_" + str(category)] = []

    sensitivityList = []
    specificityList = []

    valLossList = []
    valSensitivityList = []
    valSpecificityList = []
    valAccList = []
    valMatrixList = []

    numBatchesTotal = 0
    numUpdates = 0

    for epoch in range(int(numEpochs)):

        
        loop = tqdm(total=len(trainLoader), position=0, leave=False)


        if (epoch % stepInterval == 0) or (epoch == 0):

            numParameters = 0
            for param in model.parameters():
                numParameters += 1
                
            threshold = numParameters - ((numStepsTaken  + 1) * (numParameters / numSteps))
            print()
            print()
            print()
            print()
            print()
            print("calculated percent")
            print(1 - (threshold / numParameters))
            print("target percet")
            print(numStepsTaken / numSteps)
            numFree = 0
            currentParam = 0
            for param in model.parameters():
                if currentParam > threshold:
                    param.requires_grad=True
                    numFree += 1
                currentParam += 1

            print("percent free: " + str(numFree / numParameters))
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            numStepsTaken += 1

        for bathNum, (x, y) in enumerate(trainLoader):
                        
            if cuda:
                x = x.cuda()
                y = y.cuda()

            yHat = model(x)
            
            predCat = torch.argmax(yHat, dim=1)

            acc = torch.sum(predCat == y).item() / y.shape[0]

            if labelSmooth:
                # note that if labelSmoothing = True and the loss is something like BCE, it will throw an error
                smoothY = smoothLabel(y, numCategories, alpha = 0.2, cuda=cuda)
            else:
                smoothY = y
        
            loss = lossFunction(yHat, smoothY)
            lossList.append(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if numBatchesTotal % evaluationInterval == 0:
                model.eval()
                meanVLoss, sensitivities, specificities, accuracy, confusionMatrix = evaluateModel(model, valLoader,lossFunction, sensitivityFunction, specificityFunction, cuda, numCategories, labelSmooth)

                for category in range(numCategories):
                    sensitivitiesDict["category_" + str(category)].append((numBatchesTotal, sensitivities[category]))
                    specificitiesDict["category_" + str(category)].append((numBatchesTotal, specificities[category]))

                valLossList.append((numBatchesTotal, meanVLoss))
                valAccList.append((numBatchesTotal, accuracy))
                valMatrixList.append((numBatchesTotal, confusionMatrix))

                model.train()

            numBatchesTotal += 1
            if cuda:
                memory = torch.cuda.memory_allocated(0) / 1e9
                loop.set_description(f"_epoch:{epoch}, t_loss:{loss}, v_loss:{meanVLoss}, mem:{memory}, acc:{acc}") 
            else:
                loop.set_description(f"_epoch:{epoch}, t_loss:{loss}, v_loss:{meanVLoss}") 

            loop.update()
            gc.collect()

    

    return lossList, sensitivityList, specificityList, valLossList, sensitivitiesDict, specificitiesDict, valAccList, valMatrixList
