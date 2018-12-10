from collections import OrderedDict

import torch
from torch import nn
from torchvision import models

from test_model_pytorch_facebook_challenge import publish_evaluated_model, calc_accuracy


def load_model(checkpoint_path):
    """
    Sample code for loading a saved model
    :param checkpoint_path:
    :return:
    """
    chpt = torch.load(checkpoint_path)
    pretrained_model = getattr(models, chpt['arch'])
    if callable(pretrained_model):
        model = pretrained_model(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture not recognized")
    model.class_to_idx = chpt['class_to_idx']
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(chpt['state_dict'])
    return model


model = load_model('classifier.pth')

#calc_accuracy(model, input_image_size=224)
publish_evaluated_model(model, input_image_size=224,  username="@Gabriele.Picco", model_name="VGG19", optim="Adam",
                        criteria="NLLLoss", scheduler="StepLR", epoch=10)
