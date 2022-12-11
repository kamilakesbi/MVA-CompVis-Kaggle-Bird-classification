import torch 

import torch.nn as nn
import torchvision 
import timm 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def EfficientNet_transfer(drop1, drop2): 

  model = torchvision.models.efficientnet_b7(pretrained=True)
  for param in model.parameters():
      param.requires_grad = False
  model.classifier = nn.Sequential(
    nn.Linear(in_features=2560, out_features=1200, bias=True), 
    nn.ReLU(), 
    nn.Dropout(drop1),
    nn.Linear(1200, 300), 
    nn.ReLU(), 
    nn.Dropout(drop1), 
    nn.Linear(300,100), 
    nn.Dropout(drop2), 
    nn.ReLU(), 
    nn.Linear(100, 20), 
    nn.Dropout(drop2))
  model.to(device)
  model = nn.DataParallel(model)

  return model 


def vit(pretrained_model, drop1, drop2): 

  model = timm.create_model(pretrained_model, pretrained=True)
  for param in model.parameters():
      param.requires_grad = False
  model.head = nn.Sequential(
    nn.Linear(in_features=768, out_features=384, bias=True), 
    nn.ReLU(), 
    nn.Dropout(drop1),
    nn.Linear(384, 100), 
    nn.ReLU(), 
    nn.Dropout(drop1), 
    nn.Linear(100,20), 
    nn.Dropout(drop2))

  model.to(device)
  model = nn.DataParallel(model)
  return model
