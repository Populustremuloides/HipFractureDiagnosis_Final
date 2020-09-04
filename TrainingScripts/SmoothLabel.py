import torch

def smoothLabel(y, numCategories, alpha, cuda=False):
    ''' implements label smoothing for an integer (category) variable y. Works across batches '''

    certainty = 1 - alpha
    uncertaintyVal = alpha / numCategories
    y = torch.unsqueeze(y, 1)
    batchSize = y.shape[0]

    if cuda:
        smooth = torch.zeros(batchSize, numCategories).cuda().scatter_(1, y, certainty)
        uncertainty = torch.ones(batchSize, numCategories).cuda() * uncertaintyVal
    else:
        smooth = torch.zeros(batchSize, numCategories).scatter_(1, y, certainty)
        uncertainty = torch.ones(batchSize, numCategories) * uncertaintyVal

    smooth = smooth + uncertainty

    return smooth



#vec = torch.tensor([1,2,0,3,3,4])
#print(smoothLabel(vec, 5, 0.1))

#print(vec)
