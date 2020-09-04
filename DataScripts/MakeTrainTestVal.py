''' This file has functions that split data folders into train, test, and validation folders '''

import random
import os
from shutil import copyfile
import PIL.ImageOps as ImageOps
import PIL.Image as Image

def padding(img,expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width //2
    pad_height = delta_height //2
    padding = (pad_width,pad_height,delta_width-pad_width,delta_height-pad_height)
    return ImageOps.expand(img, padding)

def makeTrainTestVal(inputFolders):

    ''' randomly splits the jpg folders in 'inDirectory' into test, train, and validation folders
        -- also, pads images in the process so that they are all the same size -- '''

    inDirectory, valDirectory, testDirectory, trainDirectory = inputFolders
    for filename in os.listdir(inDirectory):
        if filename.endswith(".jpg"):
            sourcePath = inDirectory + "/" + filename
            num = random.randint(0, 9)

            if num == 0:
                targetPath = valDirectory + "/" + filename
            elif num == 1:
                targetPath = testDirectory + "/" + filename
            else:
                targetPath = trainDirectory + "/" + filename

            copyfile(sourcePath, targetPath)
            img = Image.open(targetPath)
            paddedImage = padding(img, 4800)
            paddedImage.save(targetPath)

def getSubGroup(folderName):

    ''' generate the names of the data folders '''

    ''' the following are the various directories (already existing in the file system) 
    that we will be copying from (inDirectory) and to (val, test, and trainDirectories) '''
    

    inDirectory = "/home/sethbw/Documents/FourierResNet/DeepLearningRadiographs/Combined"
    valDirectory = "/home/sethbw/Documents/FourierResNet/DeepLearningRadiographs/Validation"
    testDirectory = "/home/sethbw/Documents/FourierResNet/DeepLearningRadiographs/Test"
    trainDirectory = "/home/sethbw/Documents/FourierResNet/DeepLearningRadiographs/Train"

    inDirectoryFN = inDirectory + folderName
    valDirectoryFN = valDirectory + folderName
    testDirectoryFN = testDirectory + folderName
    trainDirectoryFN = trainDirectory + folderName

    return [inDirectoryFN, valDirectoryFN, testDirectoryFN, trainDirectoryFN]


inputFolders = getSubGroup("/Left_Fracture")
makeTrainTestVal(inputFolders)

inputFolders = getSubGroup("/Left_Non-Fracture")
makeTrainTestVal(inputFolders)

inputFolders = getSubGroup("/Right_Fracture")
makeTrainTestVal(inputFolders)

inputFolders = getSubGroup("/Right_Non-Fracture")
makeTrainTestVal(inputFolders)
