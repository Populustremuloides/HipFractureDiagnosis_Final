import torch
import torch.nn as nn
import numpy as np
 
class GetSpecificities(nn.Module):
    def __init__(self):
        super(GetSpecificities, self).__init__()

    def categorySpecificity(self, category, yHat, y):
        ''' calculates one vs. all version of specificity, for the given category (an int) '''

        relevantElements = (y != category)
        trueNegatives = (yHat != category) * relevantElements 
        
        if torch.sum(relevantElements) == 0:
            specificity = 0
        else:
            specificity = (torch.sum(trueNegatives).item() / torch.sum(relevantElements).item())

        return specificity

    def forward(self, yHat, y, numCategories):

        categorySpecificities = []
        for i in range(numCategories):
            specificity = self.categorySpecificity(i, yHat, y)
            categorySpecificities.append(specificity)
        return categorySpecificities


