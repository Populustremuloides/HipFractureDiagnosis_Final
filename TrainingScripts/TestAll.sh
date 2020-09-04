#!/bin/bash
# A simple script

#python3.7 TrainModel2.py -m "resnet152" -e 10 -b 10 -d augmentedImagesPath.txt -i 2 -o /home/sethbw/Documents/HipFractureDiagnosis/TrainOutputBroadTest

python3.7 TrainModel2.py -m "googlenet" -e 10 -b 100 -d augmentedImagesPath.txt -i 2 -o /home/sethbw/Documents/HipFractureDiagnosis/TrainOutputBroadTest

#python3.7 TrainModel2.py -m "alex" -e 10 -b 10 -d augmentedImagesPath.txt -i 2 -o /home/sethbw/Documents/HipFractureDiagnosis/TrainOutputBroadTest

#python3.7 TrainModel2.py -m "densenet201" -e 10 -b 10 -d augmentedImagesPath.txt -i 2 -o /home/sethbw/Documents/HipFractureDiagnosis/TrainOutputBroadTest

#python3.7 TrainModel2.py -m "wideResNet101" -e 10 -b 10 -d augmentedImagesPath.txt -i 2 -o /home/sethbw/Documents/HipFractureDiagnosis/TrainOutputBroadTest

#Qpython3.7 TrainModel2.py -m "inception" -e 10 -b 10 -d augmentedImagesPath.txt -i 2 -o /home/sethbw/Documents/HipFractureDiagnosis/TrainOutputBroadTest

