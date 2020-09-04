import random
import os
from shutil import copyfile
import torchvision
from PIL import Image
from torchvision import transforms
import os

def resizeAndNormalizeImage(inputPath, outputPath):
    ''' resize and normalize an image '''

    img = Image.open(inputPath)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))(img)
    img = torchvision.transforms.ToPILImage()(img)
    img = torchvision.transforms.Resize((350,350))(img)
    img.save(outputPath)

def resizeAndNormalizeFolder(inputFolder, targetDirectory):

    ''' resize and normalize all the images in a folder '''

    folderType = inputFolder.split("/")[-1]
    for filename in os.listdir(inputFolder):
        if filename.endswith(".jpg"):
            inputPath = inputFolder + "/" + filename
            outputPath = targetDirectory + "/" + folderType + "/" + filename
            resizeAndNormalizeImage(inputPath, outputPath)

def initializeDirectory(root):
    lf = "/Left_Fracture"
    lnf = "/Left_Non-Fracture"
    rf = "/Right_Fracture"
    rnf = "/Right_Non-Fracture"
    
    if not os.path.exists(root):
        print("a")
        os.mkdir(root)
    if not os.path.exists(root + lf):
        print("b")
        os.mkdir(root + lf)
    if not os.path.exists(root + lnf):
        print("c")
        os.mkdir(root + lnf)
    if not os.path.exists(root + rf):
        print("d")
        os.mkdir(root + rf)
    if not os.path.exists(root + rnf):
        print("e")
        os.mkdir(root + rnf)



def getSubGroups(root):

    ''' generate the names of the data folders '''

    ''' the following are the various directories (already existing in the file system) '''

    lf = "/Left_Fracture"
    lnf = "/Left_Non-Fracture"
    rf = "/Right_Fracture"
    rnf = "/Right_Non-Fracture"

    lfFolder = root + lf
    lnfFolder = root + lnf
    rfFolder = root + rf
    rnfFolder = root + rnf
    
    return [lfFolder, lnfFolder, rfFolder, rnfFolder]

valDirectory = "/home/sethbw/Documents/FourierResNet__Depreciated/DeepLearningRadiographs/Validation"
testDirectory = "/home/sethbw/Documents/FourierResNet__Depreciated/DeepLearningRadiographs/Test"
trainDirectory = "/home/sethbw/Documents/FourierResNet__Depreciated/DeepLearningRadiographs/TrainAugmented"
trainNADirectory = "/home/sethbw/Documents/FourierResNet__Depreciated/DeepLearningRadiographs/Train"

inputFolders = getSubGroups(trainNADirectory)
targetDirectory = "/home/sethbw/Documents/HipFractureDiagnosis/DeepLearningRadiographs/TrainMediumSmall"

initializeDirectory(targetDirectory)
for folder in inputFolders:
    resizeAndNormalizeFolder(folder, targetDirectory)

inputFolders = getSubGroups(testDirectory)
targetDirectory = "/home/sethbw/Documents/HipFractureDiagnosis/DeepLearningRadiographs/TestMediumSmall"
initializeDirectory(targetDirectory)
for folder in inputFolders:
    resizeAndNormalizeFolder(folder, targetDirectory)

inputFolders = getSubGroups(valDirectory)
targetDirectory = "/home/sethbw/Documents/HipFractureDiagnosis/DeepLearningRadiographs/ValidationMediumSmall"
initializeDirectory(targetDirectory)
for folder in inputFolders:
    resizeAndNormalizeFolder(folder, targetDirectory)

inputFolders = getSubGroups(trainDirectory)
targetDirectory = "/home/sethbw/Documents/HipFractureDiagnosis/DeepLearningRadiographs/TrainAugmentedMediumSmall"
initializeDirectory(targetDirectory)
for folder in inputFolders:
    resizeAndNormalizeFolder(folder, targetDirectory)


