import random
import os
from shutil import copyfile
import torchvision
from PIL import Image
from torchvision import transforms

def copyImage(inputFolder, filename, modNum):
    ''' create a copy of the image '''
    inputPath = inputFolder + "/" + filename
    outputPath = inputFolder + "/mod_" + str(modNum) + filename
    copyfile(inputPath, outputPath)
    return outputPath

def modifyImage(imagePath):
    ''' modify an image with random affine transformations '''

    img = Image.open(imagePath)
    img = torchvision.transforms.RandomAffine(degrees=random.randint(0, 30), scale=(1.3, 1.3), translate=(0.1, 0.1), shear=1.4)(img)
    img.save(imagePath)

def augmentFolders(inputFolder, numCopies):

    ''' applies random affine transformations numCopies times to .jpg files in inputFolder to make
    training/testing more difficult for the network. Keeps original images and saves copies in inputFolder '''

    for filename in os.listdir(inputFolder):
        if filename.endswith(".jpg"):

            for i in range(numCopies):
                outputPath = copyImage(inputFolder, filename, i)
                modifyImage(outputPath)

def getSubGroupgs(root):

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

#valDirectory = "C:\\Users\\BCBrown\\Desktop\\Deep Learning Radiographs-20200127T235950Z-001\\Deep Learning Radiographs\\ValidationAugmented"
#testDirectory = "C:\\Users\\BCBrown\\Desktop\\Deep Learning Radiographs-20200127T235950Z-001\\Deep Learning Radiographs\\TestAugmented"
trainDirectory = "/home/sethbw/Documents/FourierResNet/DeepLearningRadiographs/TrainAugmented"

inputFolders = getSubGroupgs(trainDirectory)
for folder in inputFolders:
    augmentFolders(folder, 10)

