import torch.nn as nn
from torchvision.models import (MobileNet_V3_Large_Weights,mobilenet_v3_large)

def build_model(num_classes:int = 38,pretrained:bool = True):
    weights = (MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    model = mobilenet_v3_large(weights=weights)

    #this gonna replace thw 1000 output classes with the number of classes in our dataset
    input_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(input_features,num_classes)
    return model