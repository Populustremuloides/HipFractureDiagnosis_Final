import torch
import torch.nn as nn
import numpy as np
 
class GetSensitivities(nn.Module):
    def __init__(self):
        super(GetSensitivities, self).__init__()


    def categorySensitivity(self, category, yHat, y):
        ''' calculates one vs. all version of senstivity, for the given category (an int) '''

        relevantElements = (y == category) # tests aside, all elements that are positive
        truePositives = (yHat == category) * relevantElements  # true positives in our selected category

        if torch.sum(relevantElements) == 0:
            sensitivity = 0
        else:
            sensitivity = torch.sum(truePositives).item() / torch.sum(relevantElements).item()

        return sensitivity


    def forward(self, yHat, y, numCategories):

        categorySensitivities = []
        for i in range(numCategories):
            sensitivityI = self.categorySensitivity(i, yHat, y)
            categorySensitivities.append(sensitivityI)

        return categorySensitivities

